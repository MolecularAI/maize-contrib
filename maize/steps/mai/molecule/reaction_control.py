"""
This file contains the node ReactionControl in Maize and some helper function and classes to
correctly running it. This node takes a reaction library csv file as input and 
automatically creates geometries for the molecular structures involved in the reaction. 
It is based on mechanistic templates stored as sdf files. After generating the structure 
the node sends the list of molecular structures for calculations. 
The type of reaction has to be specified by the user through the 'reaction' parameter. List of available 
reactions in template_repo.py.

The ReactionControl node receives inputs in the form of a list of molecular structures 
(List of Isomer Collections), from the calculations nodes. 
Function to evaluate the status of the calculations are present, failed calculations are
resubmitted. 

Successful calculations are then stored and information is used to build potential energy surfaces of the
reaction studied.

"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Any, cast, Literal
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from rdkit import Chem
from rdkit.Chem import AllChem, FragmentMatcher

from pathlib import Path
from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, FileParameter
from maize.utilities.testing import TestRig
from maize.utilities.chem.chem import Isomer, IsomerCollection, Conformer
import maize.steps.mai.molecule.template_repo as template_repo
from maize.steps.mai.molecule.compchem_utils import (
    Structure,
    Atom,
    EntryCoord,
    Loader,
    AtomEntry,
    Atom_mass,
    ConfTag,
    check_connectivity,
    check_refined,
    check_collaps
)
from maize.steps.mai.molecule.xtb import AtomType

log = logging.getLogger("run")

def split_smiles_from_reaction(smiles: str) -> list[str]:
    """
    Split a part of reaction SMILES, e.g. reactants or products
    into components. Taking care of intra-molecular complexes

    Taken from RDKit:
    https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/DaylightParser.cpp

    :param smiles: the SMILES/SMARTS
    :return: the individual components.
    """
    pos = 0
    block_start = 0
    level = 0
    in_block = 0
    components = []
    while pos < len(smiles):
        if smiles[pos] == "(":
            if pos == block_start:
                in_block = 1
            level += 1
        elif smiles[pos] == ")":
            if level == 1 and in_block:
                in_block = 2
            level -= 1
        elif level == 0 and smiles[pos] == ".":
            if in_block == 2:
                components.append(smiles[block_start + 1 : pos - 1])
            else:
                components.append(smiles[block_start:pos])
            block_start = pos + 1
            in_block = 0
        pos += 1
    if block_start < pos:
        if in_block == 2:
            components.append(smiles[block_start + 1 : pos - 1])
        else:
            components.append(smiles[block_start:pos])
    return components


def get_chg_and_mult(smi: Chem.rdchem.Mol) -> list[int]:
    """
    Returns a list with the values for charge and multiplicities of a
    molecule from its rdkit mol obj.

    Parameters
    ----------
    smi
        molecule SMILES string

    Returns
    -------
    list[int]
        list with [charge, multiplicity] values
    """

    charge = cast(int, Chem.rdmolops.GetFormalCharge(smi))
    mult = None

    n_elec = 0
    for f in smi.GetAtoms():
        n = cast(int, f.GetNumRadicalElectrons())
        n_elec += n

    mult = n_elec + 1

    out_list = [charge, mult]

    return out_list


def smarts_id(
    input_file: str, smarts_FG: str = "[NX3;H1,H2;!$(N[#6]=[#8,#16,#15,#7]);!$(N=N)]"
) -> int:
    """
    Searches for a SMARTS pattern for specific atoms in a molecule sdf file or
    SMILES string and returns the indexes of the atoms matched.

    Parameters
    ----------
    input file
        sdf file of the molecule or SMILES string
    smarts_FG
        smarts pattern

    Returns
    -------
    int
        index of the matching atom
    """

    if ".sdf" in input_file:
        matcher = FragmentMatcher.FragmentMatcher()
        matcher.Init(smarts_FG)
        molH = Chem.MolFromMolFile(input_file, removeHs=False)
        matches = matcher.GetMatches(molH)

        if matches:
            index: int = matches[0][0]
            return index
        else:
            error_message = "Could not find any match please double check"
            raise ValueError(error_message)

    else:
        matcher = FragmentMatcher.FragmentMatcher()
        matcher.Init(smarts_FG)
        molH = Chem.AddHs(Chem.MolFromSmiles(input_file))
        AllChem.EmbedMolecule(molH)
        matches = matcher.GetMatches(molH)

        if matches:
            index = cast(int, matches[0][0])
            return index
        else:
            error_message = "Could not find any match please double check"
            raise ValueError(error_message)


def max_distance_perc(
    mol: list[NDArray[np.float32]], i_point: NDArray[np.float32], perc: int
) -> float:
    """
    Returns a specified percentage of the interatomic distance between a
    selected atom and the atom in the molecule with the maximum distance from the selected one.

    Parameters
    ----------
    mol
        list of molecular coordinates
    i_point
        selected point coordinates
    perc
        desired percentage of the max distance

    Returns
    -------
    float
        desired fraction of max distance
    """

    divid = 100 / perc
    max_dist = 0.0
    for point in mol:
        dist = cast(float, np.linalg.norm(point - i_point))

        if dist > max_dist:
            max_dist = dist

    if perc != 0:
        return float(max_dist / divid)
    else:
        raise ZeroDivisionError


def remove_closest_atom_by_label(
    mol: Structure, i_point: EntryCoord, label: str, idx_at_pt: int
) -> tuple[Structure, int]:
    """
    This function finds the closest requested element to a
    reference atom and removes it from the molecule. Readjusts the
    index of the attach point if needed.

    Parameters
    ----------
    mol
        Dictionary of template coordinates
    i_point
        3D point corresponding to the reference atom
    label
        String containing the element of the atom to remove
    idx_at_pt
        index of the attach point

    Returns
    -------
    Structure
        Structure object without the closest atom with matching label
    int
        Adjusted index
    """

    dist = 1000.0
    adjusted_idx = idx_at_pt
    new_lig_list = []

    for n, e in enumerate(mol.mol_dict):
        n_dist = cast(float, np.linalg.norm(np.array(e.coords) - np.array(i_point.coords)))

        if e.element == label and n_dist < dist:
            dist = n_dist
            if n < idx_at_pt:
                adjusted_idx = idx_at_pt - 1
        else:
            new_lig_list.append(e.to_Atom())

    new_lig = Structure(new_lig_list)

    return new_lig, adjusted_idx


def find_closest_centroid(
    mol: list[NDArray[np.float32]] | NDArray[np.float32],
    i_point: list[list[float]] | NDArray[np.float32],
    rad: float = 1.5,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    This functions finds the centre of mass between the atoms within a radius to a selected atom.

    Parameters
    ----------
    mol
        Dictionary of template coordinates
    i_point
        selected points
    rad
        size of the radius

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        Centre of mass of the atoms within radius,
        Centre of mass of the atoms outside radius
    """
    neigh1: list[list[np.float32]] = []
    neigh2: list[list[np.float32]] = []
    for e in mol:
        rf = np.linalg.norm(e - i_point)

        if rf < rad:
            neigh1.append([e[0], e[1], e[2]])

        else:
            neigh2.append([e[0], e[1], e[2]])

    neighbours1 = np.array([np.array(x) for x in neigh1])

    neighbours2 = np.array([np.array(x) for x in neigh2])
    if len(neighbours2) < 2:
        neighbours2 = neighbours1

    close_centroid = np.array(
        (
            sum(neighbours1[:, 0]) / len(neighbours1[:, 0]),
            sum(neighbours1[:, 1]) / len(neighbours1[:, 1]),
            sum(neighbours1[:, 2]) / len(neighbours1[:, 2]),
        )
    )
    far_centroid = np.array(
        (
            sum(neighbours2[:, 0]) / len(neighbours2[:, 0]),
            sum(neighbours2[:, 1]) / len(neighbours2[:, 1]),
            sum(neighbours2[:, 2]) / len(neighbours2[:, 2]),
        )
    )
    return close_centroid, far_centroid


def orient_substituent(
    template_dict: list[EntryCoord],
    substituent_dict: list[EntryCoord],
    substituent_attach_center: EntryCoord,
    index: int,
) -> tuple[list[EntryCoord], list[EntryCoord], list[EntryCoord]]:
    """
    Creates ligand orientated with optimal geometry to be coordinated
    with the intermediate template. The orientation is done in order to maximize
    interatomic distances. Creates clean template and saves the atom index
    to fix in the following calculations.

    Parameters
    ----------
    template_dict
        Dictionary of template coordinates
    substituent_dict
        substituent molecule coordinates
    substituent_attach_center
        Numerical index of the attach atom on the substituent
    index
        Index of the attach atom on the template

    Returns
    -------
    tuple[list[EntryCoord], list[EntryCoord], EntryCoord]
        List of EntryCoord objects of the orientated ligand,
        List of EntryCoord objects for the template without attach point,
        EntryCoord object for the attach atom of the ligand
    """

    # get coordinates and attach point from ligand molecule object
    subst_dict = substituent_dict
    subst_attach_atom = substituent_attach_center
    attach_index = subst_dict.index(subst_attach_atom)

    subst_attach_point: NDArray[np.float32] = np.array(subst_attach_atom.coords, dtype=np.float32)
    scaled_sub_attach_atom = EntryCoord(
        element=subst_attach_atom.element,
        coords=list(
            np.array(subst_attach_atom.coords, dtype=np.float32)
            - np.array(subst_attach_atom.coords, dtype=np.float32)
        ),
    )

    keys_lig = [coord.element for coord in subst_dict]  # store atom labels
    subst_array = [
        np.array(coord.coords, dtype=np.float32) for coord in subst_dict
    ]  # access the coordinates of atoms in the substituent and turns them from list to np arrays.

    ### fix with double centroid
    unsc_close_cent = find_closest_centroid(subst_array, subst_attach_point)[
        0
    ]  # finds appropriate centroid
    subst_o = np.array([at - subst_attach_point for at in subst_array], dtype=np.float32)
    # translate substituent coordinates, by putting the attach point at the origin
    or_att_pt = unsc_close_cent - subst_attach_point
    close_subst_centroid = np.array(or_att_pt, dtype=np.float32)

    # get coordinates, labels and attach point from template molecule object
    temp_dict = template_dict  # template list of coordinates
    template_attach_atom = temp_dict[
        index
    ]  # identify the attach atom (label + coordinates) on the template based on the numerical index

    temp_attach_point = np.array(
        template_attach_atom.coords, dtype=np.float32
    )  # coordinates of the attach atom
    keys_temp = [at.element for at in temp_dict if at.element]  # store atom labels
    scaled_template_coordinates: list[list[float]] = [
        list(np.array(at.coords, dtype=np.float32) - temp_attach_point) for at in temp_dict
    ]  # center template on attach center
    scaled_temp = [
        EntryCoord(element=item1, coords=item2)
        for item1, item2 in zip(keys_temp, scaled_template_coordinates)
    ]  # recreate template molecule structure with scaled coordinates
    template_array = [
        np.array(coord.coords, dtype=np.float32) for coord in scaled_temp
    ]  # access coordinates of scaled temp as np array

    scaled_attach_pt = np.array(temp_attach_point - temp_attach_point, dtype=np.float32)
    scaled_attach_atom = EntryCoord(
        element=template_attach_atom.element,
        coords=list(scaled_attach_pt),
    )  # store label and coordinates of scaled attach point on template
    temp_centroid = find_closest_centroid(
        template_array, scaled_attach_pt, max_distance_perc(template_array, scaled_attach_pt, 10)
    )[1]

    # remove the placeholder atom on the intermediate template
    new_temp = scaled_temp
    for item in new_temp:
        if item.coords == scaled_attach_atom.coords and item.element == scaled_attach_atom.element:
            new_temp.remove(item)

    # store the Metal center of the intermediate template for future use.
    metal_cent = EntryCoord(element="", coords=[])
    for item in new_temp:
        if item.element == "Ni":
            metal_cent.element = item.element
            metal_cent.coords = item.coords
            break

    #### First rotation:  Rotates the ligand so that the metal center,
    # the attach point on the ligand and the ligand centroid will lie on the same axis.
    r = np.array(metal_cent.coords, dtype=np.float32)  # vector metal center template
    s = close_subst_centroid  # vector centroid ligand

    cross_v = np.cross(r, s).astype(
        np.float32
    )  # cross vector between two vectors, axis of rotation

    norm_cross = np.array(
        cross_v / np.sqrt(np.dot(cross_v, cross_v)), dtype=np.float32
    )  # normalised cross vector
    alfa: float = np.arccos(
        (np.dot(r, s) / (np.linalg.norm(r) * np.linalg.norm(s)))
    )  # rotation angle
    r1 = R.from_rotvec(alfa * norm_cross)  # rotation vector
    rot1 = np.array(r1.apply(subst_o), dtype=np.float32)  # rotated coordinates

    c_r_cent = find_closest_centroid(rot1, rot1[attach_index])[
        0
    ]  # coordinates of the rotated ligand

    # check the angle between the two centroids, if not zero,
    # rotates for the complementary angle and recalculate centroid
    if not np.allclose(r, c_r_cent, atol=1e-01):
        r_neg = R.from_rotvec((np.pi - alfa) * norm_cross)
        rot1 = np.array(r_neg.apply(subst_o), dtype=np.float32)
        c_r_cent = find_closest_centroid(rot1, rot1[attach_index])[0]

    #### Second rotation:
    # Rotate around the new formed axis to find the orientation that minimizes
    # the sterical hindrance. Assess different rotation angles and calculate
    # average distance for each atom from the centroid. The angle that maximizes
    # interatomic distance is assumed to be forming the best conformation.

    rotamer = np.array([])
    distance = 0.0
    rot2_angles = np.arange(0, 3.14, 0.1, dtype=np.float32)  # range of angles of rotation

    # go through a range of angles of rotation to check which is the optimal orientation.
    for beta in rot2_angles:
        ax_rot2 = (
            np.array(metal_cent.coords, dtype=np.float32) - c_r_cent
        )  # axis of rotation: vector between centroid and metal center
        r2 = R.from_rotvec(beta * ax_rot2)  # second rotation vector
        rot2 = np.array(r2.apply(rot1), dtype=np.float32)  # rotated conformer

        # check average atoms distances
        fdist = [cast(float, np.linalg.norm(k - temp_centroid)) for k in rot2]
        avg_dist = sum(fdist) / len(fdist)

        # store the rotamer only if the average distance is higher than what already stored.
        if avg_dist > distance:
            distance = avg_dist
            rotamer = rot2

    # Recreate mol dict of orientated ligand
    orientated_ligand = [EntryCoord(element=e, coords=list(c)) for e, c in zip(keys_lig, rotamer)]
    markers = [scaled_sub_attach_atom, metal_cent]
    return orientated_ligand, new_temp, markers

def make_intermediate(
    template: Structure,
    index_coord: list[int] | list[str],
    index_cov: list[int] | list[str],
    ligand_coord: str,
    ligand_coval: str,
    text: str = "default",
) -> tuple[Structure, list[int], int | None]:
    """
    This function assembles the desired intermediate structure.
    Returns the desired molecule and the atom to fix for CREST and XTB calculations.

    Parameters
    --------------
    template
        Structure of the template
    index_coord
        Indexes of the coordination attach points
    index_cov
        Indexes of the covalent attach points
    ligand_coord
        SMILES of ligand to be coordinated
    ligand_coval
        SMILES of ligand to be covalently added

    Returns
    -------
    tuple[Structure, list[int]]
        Structure of the assembled intermediate,
        Index of the atoms to fix in the sdf file
    """

    # Stores coordinates of the template taken from the input.
    # Coordinates of the final intermediate structure:
    # Initially this will contain only the template structure.
    intermediate_adduct = template.mol_dict

    # store the index of the atom in the template to fix for future calculations
    # temp_fix_indexes = len(intermediate_adduct) - len(index_coord)
    
    fix_index = []  # list of atom index to fix

    # Ligand SMILES is converted to a Structure object.
    lig_coord_rdmol = Chem.MolFromSmiles(ligand_coord)
    mol_lig_coord = Loader.molecule_from_rdkit(lig_coord_rdmol)
    lig_coord_dict = mol_lig_coord.mol_dict

    # finds index of attach point on the ligand based on SMARTS substring match.
    lig_coord_attach_pt = lig_coord_dict[smarts_id(ligand_coord, "[OX2H]")]

    # Ligand SMILES is converted to a Structure object
    lig_coval_rdmol = Chem.MolFromSmiles(ligand_coval)
    mol_lig_cov = Loader.molecule_from_rdkit(lig_coval_rdmol)
    lig_cov_dict = mol_lig_cov.mol_dict

    cov_att_idx = smarts_id(ligand_coval, "[#6][Br]")
    lig_cov_attach_pt = lig_cov_dict[cov_att_idx]
    clean_ligand_cov, adjusted_index = remove_closest_atom_by_label(
        mol_lig_cov, lig_cov_attach_pt, "Br", cov_att_idx
    )  # covalent ligand molecule object LeavingGroup removed

    clean_lig_cov_dict = clean_ligand_cov.mol_dict
    adjust_att_pt = clean_lig_cov_dict[adjusted_index]
    
    # Check for covalent interaction in the template. Attach the ligand covalently if any.
    m = 1
    reag2_idx= None
    for index in index_cov:
        if type(index) is int:
            
            istance_n, new_template, cov_markers = orient_substituent(
                intermediate_adduct, clean_lig_cov_dict, adjust_att_pt, index
            )
            intermediate_adduct = new_template
            
            
            adjust_cov_idx = 0
            if isinstance(index_coord[0], int):
                adjust_cov_idx = len(index_coord)

            for idx in index_coord:
                if isinstance(idx, int):
                    idx = idx - 1 

            cov_idx = cov_markers[0]
            met_idx = cov_markers[1]
            # readjust indexes of coordination interaction point
            # after adding ligand and removing the attach point.
            
            # append the coordinates of the ligand to the final intermediate structure
            for q in istance_n:
                intermediate_adduct.append(q)

            for indice, atm in enumerate(intermediate_adduct):
               
                if atm.coords == met_idx.coords and atm.element == met_idx.element:
                    if indice not in fix_index:
                        
                        fix_index.append(indice)
                elif atm.coords == cov_idx.coords and atm.element == cov_idx.element:
                    reag2_idx = indice - adjust_cov_idx
                    fix_index.append(reag2_idx)
                   
                    m += 1 

    # Check for coordination interactions in the template.
    n = 1  # store number of ligand added to readjust the indexes of atom.
    # This is needed to have the correct atom indexes to fix in the following calculations.
    for item in index_coord:

        if type(item) is int:
            idx_adjust = len(index_coord) - n
            istance_n, new_template, markers = orient_substituent(
                intermediate_adduct, lig_coord_dict, lig_coord_attach_pt, item
            )
            intermediate_adduct = new_template
            att_pt = markers[0]
            metal_cent = markers[1]

            # append the ligand atom coordinates to the final intermediate structure.
            for q in istance_n:
                intermediate_adduct.append(q)

            for i, coord in enumerate(intermediate_adduct):
                if coord.coords == metal_cent.coords and coord.element == metal_cent.element:
                    if i not in fix_index:
                        fix_index.append(i)
                elif coord.coords == att_pt.coords and coord.element == att_pt.element:
                    fix_index.append(
                        i - idx_adjust
                    )  # append the adjusted indexes to the list of atom index to fix.
                    n += 1
                
                    
    # create compchem_utils molecule object of the output adduct structure
    final_structure = []
    for number, line in enumerate(intermediate_adduct):
        atom = Atom(line.element, [line.coords[0], line.coords[1], line.coords[2]])
        atom.number = number
        final_structure.append(atom)

    final_mol = Structure(final_structure)
    final_mol.name = text

    # adjust the atomic index from 0-indexed to 1-indexed. For compatibility with comp chem software
    fix_index_sdf = [a + 1 for a in fix_index]
        
    if reag2_idx:
        reag2_idx =  reag2_idx + 1

    return final_mol, fix_index_sdf, reag2_idx



class ReactionControl(Node):
    """
    Takes a reaction library (CSV) row as input. Creates an instance with all
    the relevant molecule objects for the reaction.  It outputs the molecular
    structures for each species and the respective configuration files for the
    following calculations.

    """
    tags = {"chemistry", "reaction"}

    inp: Input[pd.Series] = Input()
    """ Input from experimental record as pandas Series"""

    out: Output[list[IsomerCollection]] = Output()
    """ Final output. JSON file with final calculations """

    out_crest: Output[list[IsomerCollection]] = Output()
    """ Connected to Crest node input, sends molecules for which to generate conformers """

    inp_crest: Input[list[IsomerCollection]] = Input()
    """ Connected to Crest node output, receives the generated conformers """

    out_xg16: Output[list[IsomerCollection]] = Output()
    """ Connected to xTB and Gaussian nodes input (chained), sends molecules to calculate the gibbs free energy for """

    inp_xg16: Input[list[IsomerCollection]] = Input()

    """ Input from the calculation node with calculations results to be assessed """

    reaction: Parameter[Literal["NiCatCycle_CN", "NiCatCycle_CC", "NiCatCycle_CC2",
                                "RadicalTransformations",
                                "ConditionalRadicalTransformations",
                                "Oxidize",
                                "OxidizeandReduce"]] = Parameter(default="NiCatCycle_CC2")
    """Reaction template to use"""

    _cache: dict[str, float]

    intermediate_save_location: FileParameter[Path] = FileParameter(optional=True)
    """path of folder to dump created mol objects"""


    def get_generics(self, row: pd.Series) -> None:
        """
        Stores some informations about the reaction present
        in the csv file as well as the reaction role string.

        """

        self.id = row["Libname"]  # id of the reaction library
        self.rsmi = row["rsmi_processed"]  # reaction smiles
        if (self.reaction.value == 'NiCatCycle_CC') or (self.reaction.value == 'NiCatCycle_CC2'):
            self.others = row["CorrectedReagentsSmiles"]

    @staticmethod
    def assess_crest_calculations(calc: IsomerCollection) -> tuple[bool, str]:
        """
        Checks the status of a Crest calculations and
        returns True if not finished/converged correctly.

        Removes conformers from conformers list that don't match the
        parent molecule connectivity.

        Parameters
        ----------
        calc
            The output of the calculation node

        Returns
        -------
        bool
            Boolean with status of Crest calculation
        message
            string with information about the calculation status
        """

        crest_status = True
        message = "CREST calculation failed."
        for i in range(len(calc.molecules)):
            isomer_rd_mol = calc.molecules[i]._molecule
            isomer_structure = Loader.molecule_from_rdkit(isomer_rd_mol)

            if calc.molecules[i].get_tag("crest_exit_code") == 0:
                message = "CREST finished correctly."
                n_conf = len(calc.molecules[i].conformers)
                for idx, conf in enumerate(calc.molecules[i].conformers):
                    conf_structure = Loader.molecule_from_conformer(conf)

                    if not check_connectivity(isomer_structure, conf_structure):
                        refined, clashes = check_refined(isomer_structure, conf_structure)
                        if not refined:
                            message += " Clashes from raw geometry not removed."
                            calc.molecules[i].remove_conformer(idx)
                        elif refined and clashes:
                            message += " Refined geometries but still with clashes, another optimisation needed."
                        else:
                            message += " Connectivity is changed because of optimised clashes."


                if len(calc.molecules[i].conformers) > 0:
                    n_conf_removed = n_conf - len(calc.molecules[i].conformers)
                    crest_status = False
                    message += f" {n_conf_removed} conformer(s) were removed from {calc.molecules[0].get_tag('component')} for connectivity errors."
                else:
                    message += f" However, all conformers were removed from {calc.molecules[0].get_tag('component')} due to connectivity errors."
        return crest_status, message

    @staticmethod
    def assess_xtb_calculations(calc: IsomerCollection) -> tuple[bool, str]:
        """
        Checks the status of a XTB calculations and
        returns True if not finished/converged correctly.

        Removes conformers from tag list that don't match the
        parent molecule connectivity.

        Parameters
        ---------------------------
        calc
            The output of the calculation node

        Returns
        -------
        bool
            Boolean with status of xtb calculation
        """

        xtb_status = True
        
        message = "XTB calculation failed."
        for i in range(len(calc.molecules)):
            isomer_obj = calc.molecules[i]              
            isomer_exit_codes = json.loads(cast(str, isomer_obj.get_tag("XTB_exit_codes")))

            if 0 in isomer_exit_codes.values():
                message = "XTB finished correctly."
                isomer_rd_mol = isomer_obj._molecule
                isomer_structure = Loader.molecule_from_rdkit(isomer_rd_mol)
                
                conf_geometries_tag = json.loads(cast(str, isomer_obj.get_tag("XTB_geometries")))
                n_conf = len(conf_geometries_tag)
                message += f" There are this many conformations: {n_conf}. "
                connectivity_string = f"{isomer_obj.get_tag('component')}\n"
                for idx in range(len(conf_geometries_tag)):
                    xtb_conf_json = conf_geometries_tag[str(idx)]
                    xtb_conf_structure = Loader.molecule_from_json(xtb_conf_json, f"conformer-{idx}")

                    parent_conf = calc.molecules[i].conformers[idx]
                    parent_conf_structure = Loader.molecule_from_conformer(parent_conf)
                    
                    if isomer_obj.has_tag('ar_carbon'):
                        
                        if check_collaps(xtb_conf_structure, cast(int, isomer_obj.get_tag('ar_carbon')), cast(int, isomer_obj.get_tag('alk_carbon'))):
                            
                            message += f"conformer-{idx} of isomer {isomer_obj.get_tag('component')} collapsed to product"
                            del conf_geometries_tag[str(idx)]
                        


                    if not check_connectivity(parent_conf_structure, xtb_conf_structure):
                        message += f"Warning conformer-{idx} has changed connectivity compared to parent molecule. Please double check."
                               
                if len(conf_geometries_tag) > 0:
                    n_conf_removed = n_conf - len(conf_geometries_tag)
                    xtb_status = False
                    message += f" {isomer_obj.get_tag('component')} {n_conf_removed} conformer(s) were removed for connectivity errors."
               
                else:
                    message += f" However, all conformers for {isomer_obj.get_tag('component')} were removed due to connectivity errors. See connectivity:\n{connectivity_string}"

        return xtb_status, message

    @staticmethod
    def assess_g16_calculations(calc: IsomerCollection) -> tuple[bool, str]:
        """
        Checks the status of a G16 calculations and
        returns True if not finished/converged correctly.

        Parameters
        ---------------------------
        calc
            The output of the calculation node

        Returns
        -------
        bool
            Boolean with status of g16 calculation
            True if it failed, False if it was succesful
        """

        g16_status = True
        message = f"G16 calculation failed. exit codes:"

        for i in range(len(calc.molecules)):
            isomer_exit_codes = json.loads(cast(str, calc.molecules[i].get_tag("g16_sp_exit_codes")))

            if 0 in isomer_exit_codes.values():
                
                g16_status = False
                message = "G16 finished correctly."
            else:
                message += f" {isomer_exit_codes.values()}."

        return g16_status, message

    def run(self) -> None:
        row = self.inp.receive()
        self.logger.info(f"type(row):{type(row)} row:{row}")
        self.get_generics(row)

         
        selected_reaction = getattr(template_repo, self.reaction.value)
        if hasattr(self, 'others'):
            reaction = selected_reaction(self.id, self.rsmi, roles=self.others, logger=self.logger, intermediate_save_location = self.intermediate_save_location)
            self.logger.debug(f"Extra component of the reaction are: {self.others}")
        else:
            reaction = selected_reaction(self.id, self.rsmi, logger=self.logger, intermediate_save_location = self.intermediate_save_location)
        
        self.logger.info(f"Requested reaction mechanism for {self.reaction.value}")

        list_output = []
        for component in reaction.return_reaction_components():
            self.logger.info(component.component)
            list_output.append(component.to_isomer())

        final_out = []
        for iso_list in list_output:
            for iso in iso_list:
                final_out.append(IsomerCollection([iso]))

        self.logger.info(final_out)

        for isomercollection in final_out:
            self.logger.info(
                f"Molecular Weight of {isomercollection.molecules[0].get_tag('component')} is {isomercollection.molecules[0].get_tag('mw')} "
            )
        self.logger.info(f"final_out")
        self.out_crest.send(final_out)

        crest_available_result_list = (
            self.inp_crest.receive()
        )  # These are results from Crest that we need to verify have completed succesfully
        self.logger.info(
            f"Received list of molecules from Crest node: {crest_available_result_list[0].molecules[0].name}"
        )
        crest_verified_complete: list[IsomerCollection] = []
        self.logger.info(
            f"{len(crest_available_result_list)} results to be verified for correctness"
        )


        self.out_crest.send(final_out)

        crest_results = self.inp_crest.receive()
        crest_completed: list[IsomerCollection] = []
        self.logger.info(f"{len(crest_results)} to be submitted to CREST node")

        n = 0
        while crest_results and n < 1:
            crest_to_resub = []
            crest_to_delete = []
            n += 1
            for crest_calc in crest_results:
                self.logger.info(
                    f"The isomer for {crest_calc.molecules[0].get_tag('component')} has {len(crest_calc.molecules[0].conformers)} conformers"
                )
                isomer_calc = crest_calc.molecules[0]
                if isomer_calc.has_tag('ar_carbon'):
                    self.logger.info("This is an intermediate to check!")
                    best_conf = Loader.molecule_from_conformer(isomer_calc.conformers[0])
                    self.logger.info(check_collaps(best_conf, cast(int,isomer_calc.get_tag('ar_carbon')), cast(int, isomer_calc.get_tag('alk_carbon'))))
                    if check_collaps(best_conf, cast(int, isomer_calc.get_tag('ar_carbon')), cast(int, isomer_calc.get_tag('alk_carbon'))):
                        self.logger.info(f"Ar: {isomer_calc.get_tag('ar_carbon')}, ALK: {isomer_calc.get_tag('alk_carbon')}")
                        crest_to_delete.append(crest_calc)
                        
                        self.logger.info(f"{crest_calc.molecules[0].get_tag('component')} collapsed to product. Discarded.")
                        continue
                    else:
                        self.logger.info(f"Did not find any collapsed structures for {crest_calc.molecules[0].get_tag('component')}.")   
                
            
                status, message = self.assess_crest_calculations(crest_calc)
                if status:
                    crest_to_resub.append(crest_calc)
                    self.logger.info(message)
                    self.logger.info(
                        f"The isomer has to be resubmitted, {len(crest_calc.molecules[0].conformers)} conformers left."
                    )
                    self.logger.info(f"{crest_calc} is resubmitted for iteration number {n}.")
                else:
                    crest_verified_complete.append(crest_calc)
                    self.logger.info(message)

                    self.logger.info(f"CREST for {crest_calc.molecules[0].get_tag('component')} is completed at iteration number {n}")
                    self.logger.info(
                        f"Correct calc, {crest_calc.molecules[0].get_tag('component')} has {len(crest_calc.molecules[0].conformers)} conformers"
                    )

            if crest_to_resub:
                self.out_crest.send(crest_to_resub)
                crest_results = self.inp_crest.receive()
                self.logger.info(f"{len(crest_results)} to resub to CREST")
            else:
                crest_available_result_list = []
                self.logger.info("Nothing to resubmit to CREST")

        self.logger.info(f"List of completed crest results: {crest_verified_complete}")
        self.out_xg16.send(crest_verified_complete)

        xg16_available_result_list = (
            self.inp_xg16.receive()
        )  # These are results from xTB and G16 that we need to verify have completed succesfully
        self.logger.info(
            f"Received list of molecules from xTB and Gaussian node: {xg16_available_result_list[0].molecules[0].name}"
        )
        xg16_verified_complete: list[IsomerCollection] = []
        self.logger.info(
            f"{len(xg16_available_result_list)} results to be verified for correctness"
        )

        xg16_results: list[IsomerCollection] = self.inp_xg16.receive()
        self.logger.info(f"received initial list of structures from previous nodes")
        xg16_completed: list[IsomerCollection] = []
        self.logger.info(f"{len(xg16_results)} to be submitted to XTB and G16 nodes")

        m = 0
        while xg16_results:
            xg16_to_resub = []
            
            self.logger.info(f"calling the XTB-G16 sequence for round number {m + 1}.")
            self.logger.info(f"There are {len(xg16_results)} components left to converge.")
           
            for xg16_calc in xg16_results:
                status_xtb, message_xtb = self.assess_xtb_calculations(xg16_calc)
                status_g16, message_g16 = self.assess_g16_calculations(xg16_calc)

                isomer_calc = xg16_calc.molecules[0]
                
                if status_xtb and not status_g16:
                    conformers_tag = json.loads(
                        cast(str, xg16_calc.molecules[0].get_tag("XTB_geometries"))
                    )
                    n_conformers = len(conformers_tag)

                    self.logger.info(f"XTB: {message_xtb}.")
                    self.logger.info(
                        f"The {xg16_calc.molecules[0].get_tag('component')} has to be resubmitted, {n_conformers} conformers left."
                    )
                    self.logger.info(f"{xg16_calc.molecules[0].get_tag('component')} is resubmitted for iteration number {m + 1}.")
                    xg16_to_resub.append(xg16_calc)

                elif status_g16 and not status_xtb:
                    conformers_tag = json.loads(
                        cast(str, xg16_calc.molecules[0].get_tag("XTB_geometries"))
                    )
                    n_conformers = len(conformers_tag)

                    self.logger.info(f"G16: {message_g16}.")
                    self.logger.info(
                        f"The {xg16_calc.molecules[0].get_tag('component')} has to be resubmitted, {n_conformers} conformers left."
                    )
                    self.logger.info(f"{xg16_calc.molecules[0].get_tag('component')} is resubmitted for iteration number {m}.")
                    xg16_to_resub.append(xg16_calc)
                
                elif status_xtb and status_g16:
                    conformers_tag = json.loads(
                        cast(str, xg16_calc.molecules[0].get_tag("XTB_geometries"))
                    )
                    n_conformers = len(conformers_tag)

                    self.logger.info(f"XTB: {message_xtb}, G16: {message_g16}.")
                    self.logger.info(
                        f"The {xg16_calc.molecules[0].get_tag('component')} has to be resubmitted, {n_conformers} conformers left."
                    )
                    self.logger.info(f"{xg16_calc.molecules[0].get_tag('component')} is resubmitted for iteration number {m}.")
                    xg16_to_resub.append(xg16_calc)


                else:
                    xg16_verified_complete.append(xg16_calc)
                    self.logger.info(f"{message_xtb}\n {message_g16}")


            if xg16_to_resub and m < 2:
                self.out_xg16.send(xg16_to_resub)
                
                xg16_results = self.inp_xg16.receive()
                id_to_resub: list[str] = []
                for comps in xg16_results:
                    id_to_resub.append(str(comps.molecules[0].get_tag('component')))
                self.logger.info(f"received structures to re-run for: {', '.join(id_to_resub)}")
            
            elif xg16_to_resub and m >= 2:
                self.logger.info('Too many iterations. Loop will stop here. Please check the structures.')
                xg16_results = []
            else:
                xg16_results = []
                self.logger.info("no calc to resubmit to XTB-G16 nodes")
            m +=1


        self.out.send(xg16_completed)



class TestSuiteReactionControl:
    @staticmethod
    def rdmol_to_jsontag(mol: Chem.rdchem.Mol) -> ConfTag:
        molecule = Loader.molecule_from_rdkit(mol)

        atoms: list[AtomEntry] = []
        for atom in molecule.atoms:
            atoms.append({"element": atom.label, "atom_id": atom.number, "coords": atom.position})

        conf_tag: ConfTag = {"atoms": atoms, "energy": 0.005, "gradient": 0.000}

        return conf_tag

    def test_full_wf(
        self,
        test_config: Any,
    ) -> None:
        rig = TestRig(ReactionControl, config=test_config)

        data = {
            "Libname": "EN13003-85",
            "Row": "0",
            "yield": "0.17733",
            "rsmi_processed": "Cc1ccc2c(Br)ccc(NC(=O)c3ccc(OC(C)C)cc3)c2n1.BrC1CCCC1>>Cc1ccc2c(C3CCCC3)ccc(NC(=O)c3ccc(OC(C)C)cc3)c2n1",
            "CorrectedReagentsSmiles": "(O=C([O-])[O-].[Na+].[Na+]).CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.C[Si](C)(C)[SiH]([Si](C)(C)C)[Si](C)(C)C.(CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.F[P-](F)(F)(F)(F)F.Fc1cc(F)c(-c2ccc(C(F)(F)F)cn2)c([Ir+]c2cc(F)cc(F)c2-c2cc(C(F)(F)F)ccn2)c1).(COCCOC.Cl[Ni]Cl).COCCO",
        }

        input_reac = pd.Series(data)

        rd_mol = Chem.MolFromSmiles("OC(C)C")
        rd_mol = Chem.AddHs(rd_mol)
        AllChem.EmbedMolecule(rd_mol)

        correct = IsomerCollection([Isomer.from_rdmols([rd_mol])])

        rd_mol2 = Chem.MolFromSmiles("OC(C)C")
        rd_mol2 = Chem.AddHs(rd_mol2)
        AllChem.EmbedMolecule(rd_mol2)
        wrong = IsomerCollection([Isomer.from_rdmols([rd_mol2])])

        correct_conf1 = Chem.MolFromSmiles("OC(C)C")
        correct_conf1 = Chem.AddHs(correct_conf1)
        AllChem.EmbedMolecule(correct_conf1)

        wrong_conf2 = Chem.MolFromSmiles("OCCC")
        wrong_conf2 = Chem.AddHs(wrong_conf2)
        AllChem.EmbedMolecule(wrong_conf2)

        # create isomer and add a conformer for first test_molecule
        correct.molecules[0].set_tag('component', "Correct Molecule")
        correct.molecules[0].set_tag("crest_exit_code", 0)
        correct.molecules[0].clear_conformers()
        correct.molecules[0].add_conformer(
            Conformer.from_rdmol(
                correct_conf1, parent=correct.molecules[0], renumber=False, sanitize=False
            )
        )
        correct.molecules[0].set_tag("XTB_exit_codes", '{"0" : 0}')
        correct.molecules[0].set_tag("g16_exit_codes", '{"0" : 0}')
        correct.molecules[0].set_tag(
            "XTB_geometries",
            json.dumps({"0": [TestSuiteReactionControl.rdmol_to_jsontag(correct_conf1)]}),
        )

        wrong.molecules[0].set_tag('component', "Wrong Molecule")
        wrong.molecules[0].set_tag("crest_exit_code", 0)
        wrong.molecules[0].clear_conformers()
        wrong.molecules[0].add_conformer(
            Conformer.from_rdmol(
                wrong_conf2, parent=wrong.molecules[0], renumber=False, sanitize=False
            )
        )
        wrong.molecules[0].set_tag("XTB_exit_codes", '{"0" : 0}')
        wrong.molecules[0].set_tag("g16_exit_codes", '{"0" : 0}')
        wrong.molecules[0].set_tag(
            "XTB_geometries",
            json.dumps({"0": [TestSuiteReactionControl.rdmol_to_jsontag(wrong_conf2)]}),
        )

        res = rig.setup_run(
            inputs={
                "inp": [input_reac],
                "inp_crest": [[wrong], [correct]],
                "inp_xg16": [[wrong], [correct]],
            }
        )

        corr_out = res["out"].get()
        to_crest = res["out_crest"].get()
        to_xg16 = res["out_xg16"].get()

        assert corr_out is not None
        assert to_crest is not None
        assert to_xg16 is not None

        assert len(to_crest) == 16
        for mol in to_crest:
            assert mol.molecules[0].get_tag("mw") is not "0.0"

        assert len(to_xg16) == 1
        for mol in to_xg16:
            assert mol.molecules[0].get_tag("crest_exit_code") == 0
            assert mol.molecules[0].get_tag('component') == "Correct Molecule"

        assert len(corr_out) == 1
        for mol in corr_out:
            assert mol.molecules[0].get_tag('component') == "Correct Molecule"
            assert json.loads(mol.molecules[0].get_tag("XTB_exit_codes"))["0"] == 0
            assert json.loads(mol.molecules[0].get_tag("g16_exit_codes"))["0"] == 0
