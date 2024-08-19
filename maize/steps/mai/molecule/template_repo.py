"""
This file represents the repository of Reaction classes supported (so far) for the 
ReactionControl workflows in Maize and some helper function and classes to correctly 
perform the required operations on the chemical structures.

Supported so far:

NiCatCycle_CN
    Nickel catalytic cycle for C-N cross coupling reactions. Contains structural
    templates of reaction intermediates.

NiCatCycle_CC
    Nickel catalytic cycle for C-C cross coupling reactions. Contains structural
    templates of reaction intermediates.

RadicalsTransformations
    Series of reactions for addition of radicals from NHPI esters to Michael Acceptors.

For details see documentation of specific classes.
    
"""


import numpy as np
import io
import copy
import logging
import pickle
from typing import cast, Literal
from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R

from rdkit import Chem
from rdkit.Chem import AllChem, FragmentMatcher, Draw

from maize.core.interface import Parameter, FileParameter
from maize.utilities.chem.chem import Isomer
from maize.steps.mai.molecule.compchem_utils import (
    Structure,
    Atom,
    EntryCoord,
    Loader
)

log = logging.getLogger("run")


def get_chg_and_mult(smi: Chem.rdchem.Mol) -> list[int]:
    """
    Returns a list with the values for charge and multiplicities of a
    molecule from its SMILES string.

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
) -> tuple[list[EntryCoord], list[EntryCoord], EntryCoord]:
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
        List of EntryCoord objectsd of the orientated ligand,
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

    return orientated_ligand, new_temp, scaled_sub_attach_atom


def make_intermediate(
    template: Structure,
    index_coord: list[int] | list[str],
    index_cov: list[int] | list[str],
    ligand_coord: str,
    ligand_coval: str,
    text: str = "default",
) -> tuple[Structure, list[int]]:
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
    template_mol = template.mol_dict

    # Coordinates of the final intermediate structure:
    # Initially this will contain only the template structure.
    intermediate_adduct = template_mol

    # store the index of the atom in the template to fix for future calculations
    temp_fix_indexes = len(intermediate_adduct) - len(index_coord)
    fix_index = list(range(temp_fix_indexes))  # list of atom index to fix

    # Ligand SMILES is converted to a Structure object.
    lig_coord_rdmol = Chem.MolFromSmiles(ligand_coord)
    mol_lig_coord = Loader.molecule_from_rdkit(lig_coord_rdmol)
    lig_coord_dict = mol_lig_coord.mol_dict

    # finds index of attach point on the ligand based on SMARTS substring match.
    lig_coord_attach_pt = lig_coord_dict[smarts_id(ligand_coord, "[OX2H]")]

    # Working on covalent
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
    for index in index_cov:
        if type(index) is str:
            pass
        if type(index) is int:
            istance_n, new_template, marker = orient_substituent(
                intermediate_adduct, clean_lig_cov_dict, adjust_att_pt, index
            )
            intermediate_adduct = new_template

            for i in index_coord:
                if isinstance(i, int):
                    i = i - 1

            # readjust indexes of coordination interaction point
            # after adding ligand and removing the attach point.

            # append the coordinates of the ligand to the final intermediate structure
            for q in istance_n:
                intermediate_adduct.append(q)

    # Check for coordination interactions in the template.
    n = 1  # store number of ligand added to readjust the indexes of atom.
    # This is needed to have the correct atom indexes to fix in the following calculations.
    for item in index_coord:
        if type(item) is str:
            pass

        if type(item) is int:
            idx_adjust = len(index_coord) - n
            istance_n, new_template, marker = orient_substituent(
                intermediate_adduct, lig_coord_dict, lig_coord_attach_pt, item
            )
            intermediate_adduct = new_template

            # append the ligand atom coordinates to the final intermediate structure.
            for q in istance_n:
                intermediate_adduct.append(q)

            for i, coord in enumerate(intermediate_adduct):
                if coord.coords == marker.coords and coord.element == marker.element:
                    fix_index.append(
                        i - idx_adjust
                    )  # append the adjusted indexes to the list of atom index to fix.
                    n += 1

    # create compchem_utils molecule object of the output adduct structure
    final_structure = []
    for line in intermediate_adduct:
        atom = Atom(line.element, [line.coords[0], line.coords[1], line.coords[2]])
        final_structure.append(atom)

    final_mol = Structure(final_structure)
    final_mol.name = text

    # adjust the atomic index from 0-indexed to 1-indexed. For compatibility with comp chem software
    fix_index_sdf = [a + 1 for a in fix_index]

    return final_mol, fix_index_sdf



@dataclass
class ReactionComponent:

    """
    Class type for storing molecule objects and relative metadata for calculations.
    ---
    Attributes:
    smiles
        Stores the SMILES string of the molecule.
    structure
        The 3D-structure of the molecule saved as a Mikhail.py Structure object.
    equivalents
        A float describing the equivalents of the component in the experimental record.
    parameters
        List that contains information about charge and multiplicity of the molecule.
    calc_type
        string indicating the type of calculations to perform:
        gs (Ground State), ts (Transition State).
    """

    component: str
    smiles: str = ""
    smarts: str = ""
    structure: list[Structure] = field(default_factory=list)
    mw: float = 0.0
    constraints: list[list[int]] | None = None
    equivalents: str = "0.0"
    parameters: list[int] = field(default_factory=lambda: [0, 1])
    calc_type: Literal["ts", "gs"] = "gs"

    def to_isomer(self) -> list[Isomer]:
        """
        The function takes a ReactionComponent object as an input and returns a list
        of Isomer objects corresponding to that ReactionComponent.
        Parameters of ReactionComponents are converted to Isomer tags.

        Returns
        -------
        list[Isomer]
            List of output Isomer objects
        """
        list_isomers = []
        list_mk_mol = self.structure

        for i, mk_mol in enumerate(list_mk_mol):
            # build xyz_block for each Structure object
            number = len(mk_mol.atoms)
            if not mk_mol.name:
                raise NameError(
                    "Name of the file not found. Code cannot search the file without it"
                )
            component_name = mk_mol.name
            mw = mk_mol.get_mw()

            xyz_buffer = io.StringIO()
            xyz_buffer.write(str(number) + "\n" + str(mk_mol.name) + "\n")
            for j in mk_mol.atoms:
                xyz_buffer.write(
                    j.label
                    + " "
                    + str("{0:.5f}".format(j.position[0]))
                    + " "
                    + str("{0:.5f}".format(j.position[1]))
                    + " "
                    + str("{0:.5f}".format(j.position[2]))
                    + "\n"
                )
            xyz_block = xyz_buffer.getvalue()
            xyz_buffer.close()

            # converts the xyz string to an RDkit mol and then to Isomer
            rdkit_mol = Chem.rdmolfiles.MolFromXYZBlock(xyz_block)
            isomer_mol = Isomer(rdkit_mol)

            # transfer metadata from ReactionComponent to Isomer format
            isomer_mol.name = self.component
            isomer_mol.set_tag("component", component_name)
            isomer_mol.set_tag("smiles", self.smiles)
            isomer_mol.set_tag("mw", mw)
            isomer_mol.set_tag("equivalents", self.equivalents)
            isomer_mol.set_tag("parameters", self.parameters)
            isomer_mol.set_tag("calc_type", self.calc_type)

            if self.constraints:
                lc = self.constraints[i]
                isomer_mol.set_tag("constraints", lc)

            list_isomers.append(isomer_mol)

        return list_isomers



class NiCatCycle_CN:

    """
    Class type for the Nickel Catalytic Cycle. The class contains methods
    to obtain ReactionComponent object for each species and creates the
    appropriate reaction intermediates and transition states for the
    available templates.

    Parameters
    ----------
    id
        id of the reaction in the database
    roles
        string of reaction roles in the provided dataframe
    """

    def __init__(self, id: str, roles: str) -> None:
        """
        The constructor defines the roles for the reaction. These are for
        now specifically tailored for this particular reaction mechanism.
        """
        self.name = id
        self.roles = roles
        self.solvent = ReactionComponent(component="solvent")
        self.amine = ReactionComponent(component="amine")
        self.bromide = ReactionComponent(component="bromide")
        self.reagent1 = ReactionComponent(component="reagent1")
        self.reagent2 = ReactionComponent(component="reagent2")
        self.product = ReactionComponent(component="product")
        self.intermediate1 = ReactionComponent(component="intermediate1")
        self.intermediate2 = ReactionComponent(component="intermediate2")
        self.intermediate3 = ReactionComponent(component="intermediate3")
        self.intermediate4 = ReactionComponent(component="intermediate4")
        self.intermediate5 = ReactionComponent(component="intermediate5")

        self.int_geometries = ["Octahedral", "Tetrahedral", "SquarePlanar"]
        self.intermediates_temp: dict[
            str, dict[str, tuple[str, list[int], list[str] | list[int]]]
        ] = {
            "intermediate1": {
                "Octahedral": ("templates/int1_NiOct_template.sdf", [6, 5, 4, 3], ["NA"]),
                "Tetrahedral": ("templates/int1_NiTetra_template.sdf", [4, 3], ["NA"]),
                "SquarePlanar": ("templates/int1_NiSqPl_template.sdf", [4, 3], ["NA"]),
            },
            "intermediate2": {
                "Octahedral": ("templates/int2_NiOct_template.sdf", [6, 5, 4, 3, 2], ["NA"]),
                "Tetrahedral": ("templates/int2_NiTetra_template.sdf", [4, 3, 2], ["NA"]),
                "SquarePlanar": ("templates/int2_NiSqPl_template.sdf", [4, 3, 2], ["NA"]),
            },
            "intermediate3": {
                "Octahedral": ("templates/int3_NiOct_template.sdf", [19, 18, 17], ["NA"]),
                "Tetrahedral": ("templates/int3_NiTetra_template.sdf", [17], ["NA"]),
                "SquarePlanar": ("templates/int3_NiSqPl_template.sdf", [17], ["NA"]),
            },
            "intermediate4": {
                "Octahedral": ("templates/int4_NiOct_template.sdf", [19, 18, 17], [15]),
                "Tetrahedral": ("templates/int4_NiTetra_template.sdf", [17], [14]),
                "SquarePlanar": ("templates/int4_NiSqPl_template.sdf", [17], [14]),
            },
            "intermediate5": {
                "Octahedral": ("templates/int5_NiOct_template.sdf", [6, 5, 4, 3, 2], ["NA"]),
                "Tetrahedral": ("templates/int5_NiTetra_template.sdf", [4, 3, 2], ["NA"]),
                "SquarePlanar": ("templates/int5_NiSqPl_template.sdf", [4, 3, 2], ["NA"]),
            },
        }

        self.assign_roles()

    def assign_roles(self) -> None:
        """
        This function contains the logic to assign the various roles specific to the reaction.
        It populates also the intermediates by calling the the create_intermediate function.
        """
        it_reag = 1
        for a in self.roles.split(","):
            roles = a.split("|")
            smi = roles[1]
            role = roles[2]
            equivalent = roles[3]

            # AMINE
            if role == "reactant" and "Br" not in smi:
                mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(smi))
                mol_obj.name = self.amine.component + "_" + self.name

                self.amine.smiles = smi
                self.amine.structure.append(mol_obj)
                self.amine.parameters = get_chg_and_mult(Chem.MolFromSmiles(smi))
                self.amine.equivalents = equivalent

            # BROMIDE
            elif role == "reactant" and "Br" in smi:
                mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(smi))
                mol_obj.name = self.bromide.component + "_" + self.name

                self.bromide.smiles = smi
                self.bromide.structure.append(mol_obj)
                self.bromide.parameters = get_chg_and_mult(Chem.MolFromSmiles(smi))
                self.bromide.equivalents = equivalent

            # REAGENT 1
            elif role == "reagent" and it_reag == 1:
                mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(smi))
                mol_obj.name = self.reagent1.component + "_" + self.name

                self.reagent1.smiles = smi
                self.reagent1.structure.append(mol_obj)
                self.reagent1.parameters = get_chg_and_mult(Chem.MolFromSmiles(smi))
                self.reagent1.equivalents = equivalent
                it_reag += 1

            # REAGENT 2
            elif role == "reagent" and it_reag == 2:
                mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(smi))
                mol_obj.name = self.reagent2.component + "_" + self.name

                self.reagent2.smiles = smi
                self.reagent2.structure.append(mol_obj)
                self.reagent2.parameters = get_chg_and_mult(Chem.MolFromSmiles(smi))
                self.reagent2.equivalents = equivalent
                it_reag += 1

            # SOLVENT
            elif role == "solvent":
                mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(smi))
                mol_obj.name = self.solvent.component + "_" + self.name

                self.solvent.smiles = smi
                self.solvent.structure.append(mol_obj)
                self.solvent.parameters = get_chg_and_mult(Chem.MolFromSmiles(smi))
                self.solvent.equivalents = equivalent

            # PRODUCT
            elif role == "product":
                mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(smi))
                mol_obj.name = self.product.component + "_" + self.name

                self.product.smiles = smi
                self.product.structure.append(mol_obj)
                self.product.parameters = get_chg_and_mult(Chem.MolFromSmiles(smi))
                self.product.equivalents = equivalent

        # intermediate1
        self.intermediate1.structure = self.create_intermediates("intermediate1")[0]
        self.intermediate1.constraints = self.create_intermediates("intermediate1")[1]

        # intermediate2
        self.intermediate2.structure = self.create_intermediates("intermediate2")[0]
        self.intermediate2.constraints = self.create_intermediates("intermediate2")[1]

        # intermediate3
        self.intermediate3.structure = self.create_intermediates("intermediate3")[0]
        self.intermediate3.constraints = self.create_intermediates("intermediate3")[1]

        # intermediate4
        self.intermediate4.structure = self.create_intermediates("intermediate4")[0]
        self.intermediate4.constraints = self.create_intermediates("intermediate4")[1]

        # intermediate5

        self.intermediate5.structure = self.create_intermediates("intermediate5")[0]
        self.intermediate5.constraints = self.create_intermediates("intermediate5")[1]

    @staticmethod
    def retrieve_template(
        inter: dict[str, tuple[str, list[int], list[str] | list[int]]], geom: str
    ) -> tuple[Structure, list[int], list[str] | list[int]]:
        """
        This function takes as input an intermediate and a type of geometry and
        returns the respective template as Structure object and the associated
        indexes for covalent and coordinate interactions.

        Parameters
        ----------
        inter
            intermediate to retrieve
        geom
            geometry associated with the structure

        Returns
        -------
        tuple[Structure, list[int], list[int] | None]
            Structure object of the template,
            List of indexes corresponding to coordination interactions,
            List of indexes corresponding to covalent interactions
        """
        template_repo = inter

        filename = template_repo[geom][0]
        indexes_coord = template_repo[geom][1]
        index_cov = template_repo[geom][2]

        template_load = Loader(filename)
        temp_molecule = template_load.molecule()

        return temp_molecule, indexes_coord, index_cov

    def create_intermediates(
        self, intermediate_template: str
    ) -> tuple[list[Structure], list[list[int]]]:
        """
        This function creates the proper intermediate molecule
        object and the associated atom to be constrained.

        Parameters
        ----------
        intermediate_template
            The template structure for the intermediate

        Returns
        -------
         tuple[list[Structure], list[list[int]]]
            List of intermediates structures,
            List of constraints indexes for each intermediate
        """
        intermediates_list: list[Structure] = []
        constraints_list: list[list[int]] = []

        for g in self.int_geometries:
            geometry_key = self.intermediates_temp[intermediate_template]
            temp_mol, temp_indexes, cov_index = NiCatCycle_CN.retrieve_template(geometry_key, g)
            test_iter, atms_to_fix = make_intermediate(
                temp_mol,
                temp_indexes,
                cov_index,
                self.amine.smiles,
                "{}_{}_{}".format(intermediate_template, g, self.name),
            )
            intermediates_list.append(test_iter)
            constraints_list.append(atms_to_fix)

        return intermediates_list, constraints_list

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for
        each species in the reaction mechanism that has been populated

        Returns
        -------
        list[ReactionComponent]
            List of ReactionComponents objects created successfully.
        """

        available = []
        std_reaction_components = [
            self.solvent,
            self.amine,
            self.bromide,
            self.reagent1,
            self.reagent2,
            self.intermediate1,
            self.intermediate2,
            self.intermediate3,
            self.intermediate4,
            self.intermediate5,
            self.product,
        ]

        for f in std_reaction_components:
            if f.structure:
                available.append(f)

        return available


class NiCatCycle_CC:

    """
    Class type for the internally modified Nickel Catalytic Cycle for CC coupling for iLAB.
    Temporary class. To be merged with

    The class contains methods to obtain ReactionComponent object for each
    species and creates the appropriate reaction intermediates and transition
    states for the available templates.

    """

    def __init__(self, id: str, rsmi: str, roles: str) -> None:
        """
        The constructor defines the roles for the reaction. These are for
        now specifically tailored for this particular reaction mechanism.
        In the future we shall evaluate if it is possible to generalize
        most of it for different reaction. Also check if the class format
        is necessary, should consider something more versatile such as
        JSON objects.

        """
        self.name = id
        self.rsmi = rsmi
        self.others = roles
        self.solvent = ReactionComponent(component="solvent")
        self.reactant1 = ReactionComponent(component="reagent1")
        self.reactant2 = ReactionComponent(component="reagent2")
        self.product = ReactionComponent(component="product")
        self.Ni0_inter = ReactionComponent(component="Ni_intermediate1")
        self.Ni1_inter = ReactionComponent(component="Ni_intermediate2", parameters=[0, 2])
        self.Ni2_inter = ReactionComponent(component="Ni_intermediate3")
        self.Ni3_inter = ReactionComponent(component="Ni_intermediate4", parameters=[0, 2])

        self.int_geometries = ["TrigonalPlanar", "Tetrahedral", "SquarePlanar"]
        self.intermediates_temp: dict[
            str, dict[str, tuple[str, list[int] | list[str], list[str] | list[int]]]
        ] = {
            "Ni0": {
                "TrigonalPlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/Ni0_trpl_CC_856.sdf",
                    [3, 2, 1],
                    ["NA"],
                ),
                "Tetrahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/Ni0_tetra_CC_856.sdf",
                    [4, 3, 2, 1],
                    ["NA"],
                ),
                "SquarePlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/Ni0_sqpl_CC_856.sdf",
                    [4, 3, 2, 1],
                    ["NA"],
                ),
            },
            "Ni1": {
                "TrigonalPlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_trpl_CC_856.sdf",
                    [2, 1],
                    ["NA"],
                ),
                "Tetrahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_tetra_CC_856.sdf",
                    [3, 2, 1],
                    ["NA"],
                ),
                "SquarePlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_sqpl_CC_856.sdf",
                    [3, 2, 1],
                    ["NA"],
                ),
            },
            "Ni2": {
                "TrigonalPlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_trpl_CC_856.sdf",
                    [2],
                    ["NA"],
                ),
                "Tetrahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_tetra_CC_856.sdf",
                    [2, 1],
                    ["NA"],
                ),
                "SquarePlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_sqpl_CC_856.sdf",
                    [2, 1],
                    ["NA"],
                ),
            },
            "Ni3": {
                "TrigonalPlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_trpl_CC_856.sdf",
                    ["NA"],
                    [2],
                ),
                "Tetrahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_tetra_CC_856.sdf",
                    [1],
                    [2],
                ),
                "SquarePlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_sqpl_CC_856.sdf",
                    [1],
                    [2],
                ),
            },
        }

        self.assign_roles()

    def assign_roles(self) -> None:
        """
        This function contains the logic to assign the various roles specific to the reaction.
        It populates also the intermediates by calling the the create_intermediate function.

        """

        separ = self.rsmi.split(">>")
        left_hand = separ[0]
        right_hand = separ[1]

        # reactant 1
        reac1_smiles = left_hand.split(".")[0]
        reac1_rdmol = Chem.MolFromSmiles(reac1_smiles)
        mol_reac1 = Loader.molecule_from_rdkit(reac1_rdmol)
        mol_reac1.name = self.reactant1.component
        self.reactant1.smiles = reac1_smiles
        # self.reactant1.mw = mol_reac1.get_mw()
        self.reactant1.structure.append(mol_reac1)
        self.reactant1.parameters = get_chg_and_mult(reac1_rdmol)

        # reactant 2
        reac2_smiles = left_hand.split(".")[1]
        reac2_rdmol = Chem.MolFromSmiles(reac2_smiles)
        mol_reac2 = Loader.molecule_from_rdkit(reac2_rdmol)
        mol_reac2.name = self.reactant2.component
        self.reactant2.smiles = reac2_smiles
        # self.reactant2.mw = mol_reac2.get_mw()
        self.reactant2.structure.append(mol_reac2)
        self.reactant2.parameters = get_chg_and_mult(reac2_rdmol)

        # product
        prod_smiles = right_hand.split(".")[0]
        prod_rdmol = Chem.MolFromSmiles(prod_smiles)
        mol_prod = Loader.molecule_from_rdkit(prod_rdmol)
        mol_prod.name = self.product.component
        self.product.smiles = prod_smiles
        # self.product.mw = mol_prod.get_mw()
        self.product.structure.append(mol_prod)
        self.product.parameters = get_chg_and_mult(prod_rdmol)

        # solvent
        solv_smiles = self.others.split(".")[-1]
        solv_rdmol = Chem.MolFromSmiles(solv_smiles)
        mol_solv = Loader.molecule_from_rdkit(solv_rdmol)
        mol_solv.name = self.solvent.component
        self.solvent.smiles = solv_smiles
        # self.solvent.mw = mol_solv.get_mw()
        self.solvent.structure.append(mol_solv)
        self.solvent.parameters = get_chg_and_mult(solv_rdmol)

        # Ni0 inter

        self.Ni0_inter.structure = self.create_intermediates("Ni0")[0]
        self.Ni0_inter.constraints = self.create_intermediates("Ni0")[1]

        # Ni1 inter
        self.Ni1_inter.structure = self.create_intermediates("Ni1")[0]
        self.Ni1_inter.constraints = self.create_intermediates("Ni1")[1]

        # Ni2 inter
        self.Ni2_inter.structure = self.create_intermediates("Ni2")[0]
        self.Ni2_inter.constraints = self.create_intermediates("Ni2")[1]

        # Ni3 inter
        self.Ni3_inter.structure = self.create_intermediates("Ni3")[0]
        self.Ni3_inter.constraints = self.create_intermediates("Ni3")[1]

    @staticmethod
    def retrieve_template(
        int: dict[str, tuple[str, list[int] | list[str], list[str] | list[int]]], geom: str
    ) -> tuple[Structure, list[int] | list[str], list[int] | list[str]]:
        """
        This function takes as input an intermediate and a type of geometry
        and returns the respective template as Structure object and the
        associated indexes for covalent and coordinate interactions.

        """

        template_repo = int
        filename = template_repo[geom][0]
        indexes_coord = template_repo[geom][1]
        index_cov = template_repo[geom][2]

        template_load = Loader(filename)
        temp_molecule = template_load.molecule()

        return temp_molecule, indexes_coord, index_cov

    def create_intermediates(
        self, intermediate_template: str
    ) -> tuple[list[Structure], list[list[int]]]:
        """
        This function creates the proper intermediate molecule
        object and the associated atom to be constrained.

        """
        intermediates_list: list[Structure] = []
        constraints_list: list[list[int]] = []

        intermediates_list = []
        constraints_list = []

        for geom in self.int_geometries:
            temp_mol, temp_indexes, cov_index = NiCatCycle_CC.retrieve_template(
                self.intermediates_temp[intermediate_template], geom
            )

            test_iter, atms_to_fix = make_intermediate(
                temp_mol,
                temp_indexes,
                cov_index,
                self.solvent.smiles,
                self.reactant2.smiles,
                "{}_{}".format(intermediate_template, geom),
            )

            intermediates_list.append(test_iter)
            constraints_list.append(atms_to_fix)

        return intermediates_list, constraints_list

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for each
        species in the reaction mechanism that has been populated

        """

        available: list[ReactionComponent] = []
        std_reaction_components = [
            self.solvent,
            self.reactant1,
            self.reactant2,
            self.product,
            self.Ni0_inter,
            self.Ni1_inter,
            self.Ni2_inter,
            self.Ni3_inter,
        ]

        for f in std_reaction_components:
            if f.structure:
                available.append(f)

        return available

class RadicalTransformations:

    def __init__(self, id: str, rsmi: str, logger: logging.Logger, debug_save_location: FileParameter[Path]) -> None:
        """
        The constructor defines the roles for this specific radical reaction.
        Reactant1 can decompose twice to afford two different radicals. One of 
        the two radicals undergoes michael addition with Reactant2, to afford
        m_adduct.
        
        Reaction Scheme
        ---------------
        Reactant1 ---> Radical1• 
        Reactant1 ---> Radical2•
        Radical2• + Reactant2 ---> m_adduct•

        The entry structure shall generate two separate radicals one with the carboxylic group 
        and the second without. 

        The final product is the Michael addition product of the radicals
        In the future we shall evaluate if it is possible to generalize
        most of it for different reaction. Also check if the class format
        is necessary, should consider something more versatile such as
        JSON objects.

        """
        self.name = id
        self.rsmi = rsmi
        self.solvent = ReactionComponent(component="solvent")
        self.reactant1 = ReactionComponent(component="reactant", smarts="[C:1][C](=[O])[O][N]1C(=O)c2ccccc2C1=O")
        self.reactant2 = ReactionComponent(component="michael_acceptor", smarts= "[C:2]=[C:3][#6,#7,#8,#15,#16:4]")
        self.radical1 = ReactionComponent(component="radical-CCO",  parameters=[0,2])
        self.radical2 = ReactionComponent(component="radical-C", parameters=[0,2])
        self.m_adduct = ReactionComponent(component="adduct", smarts= "[C:1][C:2][C-:3][#6,#7,#8,#15,#16:4]", parameters=[0,2])
        
        self.reaction_SMARTS = f"{self.reactant1.smarts}.{self.reactant2.smarts}>>{self.m_adduct.smarts}"
        
        #### DEBUGGING STUFF BY BOB
        self.logger = logger
        self.debug_save_location = debug_save_location
        ### END OF DEBUGGING

        self.assign_roles()

    def assign_roles(self) -> None:
        """
        This function contains the logic to assign the various roles specific to the reaction.
        It populates also the radicals attrubyte by calling the specific functions.
        """
        
        ### reactant1 ###
        reactant1_smiles = self.rsmi.split('|')[0]
        self.reactant1.smiles = reactant1_smiles
        
        reactant1_rdmol = Chem.MolFromSmiles(reactant1_smiles)
        reactant1_mol_obj = Loader.molecule_from_rdkit(reactant1_rdmol)
        reactant1_mol_obj.name = self.reactant1.component + "_" + self.name
        self.reactant1.structure.append(reactant1_mol_obj)
        self.reactant1.parameters = get_chg_and_mult(reactant1_rdmol)
        
        ### reactant2 ###
        reactant2_smiles = self.rsmi.split('|')[1]
        self.reactant2.smiles = reactant2_smiles
        
        reactant2_mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(reactant2_smiles))
        reactant2_mol_obj.name = self.reactant2.component + "_" + self.name
        self.reactant2.structure.append(reactant2_mol_obj)
        self.reactant2.parameters = get_chg_and_mult(Chem.MolFromSmiles(reactant2_smiles))

        ### radical1 ###
        radical1_mol_obj = Loader.molecule_from_rdkit(self.decomposition(reactant1_rdmol, "[C:1][C:5](=[O:6])[O-:7]"))
        radical1_mol_obj.name = self.radical1.component
        self.radical1.structure.append(radical1_mol_obj)
        
        ### radical2 ###
        radical2_mol_obj = Loader.molecule_from_rdkit(self.decomposition(reactant1_rdmol, "[C:1]"))
        radical2_mol_obj.name = self.radical2.component
        self.radical2.structure.append(radical2_mol_obj)

        ### m_adduct ###
        smarts_object = AllChem.ReactionFromSmarts(self.reaction_SMARTS)
        reactant1_rdmol = Chem.MolFromSmiles(self.reactant1.smiles)
        reactant2_rdmol = Chem.MolFromSmiles(self.reactant2.smiles)
       
        ps_i = smarts_object.RunReactants((reactant1_rdmol, reactant2_rdmol))
        
        # BOBS DEBUGGING STUFF HERE
        def save_mol_image(mol_obj : Chem.rdmol.Mol, mol_name : str, filename : str) -> None:
            try:
                self.logger.info(f"Trying to save mol_obj: {mol_name} as {filename}.png")
                debug_folder = Path(self.debug_save_location.value)
                filepath = Path(debug_folder / f"{filename}.png")
                Draw.MolToFile(mol=mol_obj, legend=mol_name, filename=filepath, imageType="png")
            except Exception as e:
                self.logger.info(f"Error while trying to save mol image for: {mol_name}!")

        def save_smiles_file(smiles : str, filename : str) -> None:
            self.logger.info(f"Debug save location: {self.debug_save_location}\nvalue:{self.debug_save_location.value}")
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.txt")
            with open(filepath, "a") as file:
                file.write(smiles)

        def save_to_xyz(structure_obj : Structure, filename : str) -> None:
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.xyz")
            structure_obj.write_xyz(path=filepath)

        def save_mol_obj(mol_obj : Chem.rdmol.Mol, filename : str) -> None:
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.pk")
            with open(filepath, "wb") as file:
                pickle.dump(mol_obj, file)

        self.logger.info(f"DEBUGGING:\nreactant 1: {reactant1_smiles}")
        self.logger.info(f"DEBUGGING:\nreactant 2: {reactant2_smiles}")

        # debugging: saving the mol IMAGE files
        save_mol_image(reactant1_rdmol, reactant1_mol_obj.name, "reactant_1")
        save_mol_image(reactant2_rdmol, reactant2_mol_obj.name, "reactant_2")
        #save_mol_image(radical1_mol_obj, radical1_mol_obj.name, "radical_1")
        #save_mol_image(radical2_mol_obj, radical2_mol_obj.name, "radical_2")

        # debugging: saving smiles for those that we have (for some reason the radicals objects dont want
        # to be turned into smiles, rdkit crashes every time I try to do anything with it)
        save_smiles_file(reactant1_smiles, "reactant1_smiles")
        save_smiles_file(reactant2_smiles, "reactant2_smiles")
        save_smiles_file(self.m_adduct.smiles, "m_adduct_smiles")
        
        # debugging: saving mol objects
        save_mol_obj(reactant1_rdmol, "m_adduct_smiles")
        save_mol_obj(reactant2_rdmol, "m_adduct_smiles")
        save_mol_obj(radical1_mol_obj, "radical1")
        save_mol_obj(radical2_mol_obj, "radical2")
        save_to_xyz(radical1_mol_obj, "radical1")
        save_to_xyz(radical2_mol_obj, "radical2")


        for prod_idx in range(len(ps_i)):
            product = ps_i[prod_idx][0]
            self.logger.info(f"SMILES of m_adduct: {self.m_adduct.smiles}. Now trying to save mol image...")
            self.m_adduct.smiles = Chem.MolToSmiles(product)
            save_mol_image(product, self.m_adduct.smiles, "adduct")
        
        m_adduct_mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(self.m_adduct.smiles))
        m_adduct_mol_obj.name = self.m_adduct.component
        self.m_adduct.structure.append(m_adduct_mol_obj)
        
    
    def decomposition(self, mol: Chem.rdchem.Mol, SMARTS_decomp: str) -> Chem.rdchem.Mol:
        """
        The functions takes a SMARTS pattern and an RDkit molecule as inputs and performs 
        a decompositon of the molecule affording the radical corresponding to the 
        transformation represented in the SMARTS pattern.

        Parameters
        ----------
        mol
            RDkit molecule object for the reactant.
        SMARTS_decomp
            SMARTS string of the substructure to remove.

        Returns
        -------
        Chem.rdchem.Mol
            Molecule object of the radical product.
        """

        product_decomp = copy.deepcopy(mol)
        smarts_reactant = "[C:1][C:5](=[O:6])[O:7][N]1[C](=[O])[c]2[c][c][c][c][c]2[C]1=[O]"
        reaction_SMARTS_decomp = f"{smarts_reactant}>>{SMARTS_decomp}"
        smirks_object = AllChem.ReactionFromSmarts(reaction_SMARTS_decomp)

        smirks_object.RunReactantInPlace(product_decomp)


        return product_decomp

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for
        each species in the reaction mechanism that has been populated

        Returns
        -------
        list[ReactionComponent]
            List of ReactionComponents objects created successfully.
        """

        available = []
        std_reaction_components = [
            self.solvent,
            self.reactant1,
            self.reactant2,
            self.radical1,
            self.radical2,
            self.m_adduct
            ]

        for f in std_reaction_components:
            if f.structure:
                available.append(f)

        return available


class ConditionalRadicalTransformations1:

    def __init__(self, id: str, rsmi: str, logger: logging.Logger, debug_save_location: FileParameter[Path]) -> None:
        """
        The constructor defines the roles for this specific radical reaction.
        Reactant1 can decompose twice to afford two different radicals. One of 
        the two radicals undergoes michael addition with Reactant2, to afford
        m_adduct.
        
        Reaction Scheme
        ---------------
        Reactant1 ---> Radical1• 
        Reactant1 ---> Radical2•
        Radical2• + Reactant2 ---> m_adduct•

        The entry structure shall generate two separate radicals one with the carboxylic group 
        and the second without. 

        The final product is the Michael addition product of the radicals
        In the future we shall evaluate if it is possible to generalize
        most of it for different reaction. Also check if the class format
        is necessary, should consider something more versatile such as
        JSON objects.

        """
        self.name = id
        self.rsmi = rsmi
        self.solvent = ReactionComponent(component="solvent")
        self.reactant1 = ReactionComponent(component="reactant", smarts="[C:1][C](=[O])[O][N]1C(=O)c2ccccc2C1=O")
        self.reactant2 = ReactionComponent(component="michael_acceptor", smarts= "[C:2]=[C:3][#6,#7,#8,#15,#16:4]")
        self.alt_reactant2 = ReactionComponent(component="alt_michael_acceptor", smarts= "[N:2]=[C:3][#6:4]")
        self.radical1 = ReactionComponent(component="radical-CCO",  parameters=[0,2])
        self.radical2 = ReactionComponent(component="radical-C", parameters=[0,2])
        self.m_adduct = ReactionComponent(component="adduct", smarts= "[C:1][C:2][C-:3][#6,#7,#8,#15,#16:4]", parameters=[0,2])
        self.alt_m_adduct = ReactionComponent(component="alt_adduct", smarts = "[C:1][N:2][C-:3][#6:4]", parameters=[0,2])
        self.reaction_SMARTS = f"{self.reactant1.smarts}.{self.reactant2.smarts}>>{self.m_adduct.smarts}"
        self.alt_SMARTS = f"{self.reactant1.smarts}.{self.alt_reactant2.smarts}>>{self.alt_m_adduct.smarts}"
        
        #### DEBUGGING STUFF BY BOB
        self.logger = logger
        self.debug_save_location = debug_save_location
        ### END OF DEBUGGING

        self.assign_roles()

    def assign_roles(self) -> None:
        """
        This function contains the logic to assign the various roles specific to the reaction.
        It populates also the radicals attribute by calling the specific functions.
        """
        
        ### reactant1 ###
        reactant1_smiles = self.rsmi.split('|')[0]
        self.reactant1.smiles = reactant1_smiles
        
        reactant1_rdmol = Chem.MolFromSmiles(reactant1_smiles)
        reactant1_mol_obj = Loader.molecule_from_rdkit(reactant1_rdmol)
        reactant1_mol_obj.name = self.reactant1.component + "_" + self.name
        self.reactant1.structure.append(reactant1_mol_obj)
        self.reactant1.parameters = get_chg_and_mult(reactant1_rdmol)
        
        ### CHECK if we should use alternative ###
        reactant2_smiles = self.rsmi.split('|')[1]
        reactant2_mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(reactant2_smiles))
        check_smarts = Chem.MolFromSmarts(self.reactant2.smarts)
        reactant2_rdkit_mol = Chem.MolFromSmiles(reactant2_smiles)
        if not reactant2_rdkit_mol.HasSubstructMatch(check_smarts):
            self.reactant2 = self.alt_reactant2
            self.reaction_SMARTS = self.alt_SMARTS
        
        ### reactant2 ###
        self.reactant2.smiles = reactant2_smiles
        
        reactant2_mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(reactant2_smiles))
        reactant2_mol_obj.name = self.reactant2.component + "_" + self.name
        self.reactant2.structure.append(reactant2_mol_obj)
        self.reactant2.parameters = get_chg_and_mult(Chem.MolFromSmiles(reactant2_smiles))

        ### radical1 ###
        radical1_mol_obj = Loader.molecule_from_rdkit(self.decomposition(reactant1_rdmol, "[C:1][C:5](=[O:6])[O-:7]"))
        radical1_mol_obj.name = self.radical1.component
        self.radical1.structure.append(radical1_mol_obj)
        
        ### radical2 ###
        radical2_mol_obj = Loader.molecule_from_rdkit(self.decomposition(reactant1_rdmol, "[C:1]"))
        radical2_mol_obj.name = self.radical2.component
        self.radical2.structure.append(radical2_mol_obj)

        ### m_adduct ###
        smarts_object = AllChem.ReactionFromSmarts(self.reaction_SMARTS)
        reactant1_rdmol = Chem.MolFromSmiles(self.reactant1.smiles)
        reactant2_rdmol = Chem.MolFromSmiles(self.reactant2.smiles)
        ps_i = smarts_object.RunReactants((reactant1_rdmol, reactant2_rdmol))
        
        # BOBS DEBUGGING STUFF HERE
        def save_mol_image(mol_obj: Chem.rdmol.Mol, mol_name: str, filename: str) -> None:
            try:
                self.logger.info(f"Trying to save mol_obj: {mol_name} as {filename}.png")
                debug_folder = Path(self.debug_save_location.value)
                filepath = Path(debug_folder / f"{filename}.png")
                Draw.MolToFile(mol=mol_obj, legend=mol_name, filename=filepath, imageType="png")
            except Exception as e:
                self.logger.info(f"Error while trying to save mol image for: {mol_name}!")

        def save_smiles_file(smiles: str, filename: str) -> None:
            self.logger.info(f"Debug save location: {self.debug_save_location}\nvalue:{self.debug_save_location.value}")
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.txt")
            with open(filepath, "a") as file:
                file.write(smiles)

        def save_to_xyz(structure_obj: Structure, filename: str) -> None:
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.xyz")
            structure_obj.write_xyz(path=filepath)

        def save_mol_obj(mol_obj: Structure, filename: str) -> None:
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.pk")
            with open(filepath, "wb") as file:
                pickle.dump(mol_obj, file)

        self.logger.info(f"DEBUGGING:\nreactant 1: {reactant1_smiles}")
        self.logger.info(f"DEBUGGING:\nreactant 2: {reactant2_smiles}")

        # debugging: saving the mol IMAGE files
        save_mol_image(reactant1_rdmol, reactant1_mol_obj.name, "reactant_1")
        save_mol_image(reactant2_rdmol, reactant2_mol_obj.name, "reactant_2")
        #save_mol_image(radical1_mol_obj, radical1_mol_obj.name, "radical_1")
        #save_mol_image(radical2_mol_obj, radical2_mol_obj.name, "radical_2")

        # debugging: saving smiles for those that we have (for some reason the radicals objects dont want
        # to be turned into smiles, rdkit crashes every time I try to do anything with it)
        save_smiles_file(reactant1_smiles, "reactant1_smiles")
        save_smiles_file(reactant2_smiles, "reactant2_smiles")
        save_smiles_file(self.m_adduct.smiles, "m_adduct_smiles")
        # debugging: saving mol objects
        save_mol_obj(reactant1_rdmol, "m_adduct_smiles")
        save_mol_obj(reactant2_rdmol, "m_adduct_smiles")
        save_mol_obj(radical1_mol_obj, "radical1")
        save_mol_obj(radical2_mol_obj, "radical2")
        save_to_xyz(radical1_mol_obj, "radical1")
        save_to_xyz(radical2_mol_obj, "radical2")
        
        for prod_idx in range(len(ps_i)):
            product = ps_i[prod_idx][0]
            self.logger.info(f"SMILES of m_adduct: {self.m_adduct.smiles}. Now trying to save mol image...")
            self.m_adduct.smiles = Chem.MolToSmiles(product)
            save_mol_image(product, self.m_adduct.smiles, "adduct")
        m_adduct_mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(self.m_adduct.smiles))
        m_adduct_mol_obj.name = self.m_adduct.component
        self.m_adduct.structure.append(m_adduct_mol_obj)
        
    
    def decomposition(self, mol: Chem.rdchem.Mol, SMARTS_decomp: str) -> Chem.rdchem.Mol:
        """
        The functions takes a SMARTS pattern and an RDkit molecule as inputs and performs 
        a decompositon of the molecule affording the radical corresponding to the 
        transformation represented in the SMARTS pattern.

        Parameters
        ----------
        mol
            RDkit molecule object for the reactant.
        SMARTS_decomp
            SMARTS string of the substructure to remove.

        Returns
        -------
        Chem.rdchem.Mol
            Molecule object of the radical product.
        """

        product_decomp = copy.deepcopy(mol)
        smarts_reactant = "[C:1][C:5](=[O:6])[O:7][N]1[C](=[O])[c]2[c][c][c][c][c]2[C]1=[O]"
        reaction_SMARTS_decomp = f"{smarts_reactant}>>{SMARTS_decomp}"
        smirks_object = AllChem.ReactionFromSmarts(reaction_SMARTS_decomp)

        smirks_object.RunReactantInPlace(product_decomp)


        return product_decomp

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for
        each species in the reaction mechanism that has been populated

        Returns
        -------
        list[ReactionComponent]
            List of ReactionComponents objects created successfully.
        """

        available = []
        std_reaction_components = [
            self.solvent,
            self.reactant1,
            self.reactant2,
            self.radical1,
            self.radical2,
            self.m_adduct
            ]

        for f in std_reaction_components:
            if f.structure:
                available.append(f)

        return available

class ConditionalRadicalTransformations2:

    def __init__(self, id: str, rsmi: str, logger: logging.Logger, debug_save_location: FileParameter[Path]) -> None:
        """
        The constructor defines the roles for this specific radical reaction.
        Reactant1 can decompose twice to afford two different radicals. One of 
        the two radicals undergoes michael addition with Reactant2, to afford
        m_adduct.
        
        Reaction Scheme
        ---------------
        Reactant1 ---> Radical1• 
        Reactant1 ---> Radical2•
        Radical2• + Reactant2 ---> m_adduct•

        The entry structure shall generate two separate radicals one with the carboxylic group 
        and the second without. 

        The final product is the Michael addition product of the radicals
        In the future we shall evaluate if it is possible to generalize
        most of it for different reaction. Also check if the class format
        is necessary, should consider something more versatile such as
        JSON objects.

        """
        self.name = id
        self.rsmi = rsmi
        self.solvent = ReactionComponent(component="solvent")
        self.reactant1 = ReactionComponent(component="reactant", smarts="[C:1][C](=[O])[O][N]1C(=O)c2ccccc2C1=O")
        self.reactant2 = ReactionComponent(component="michael_acceptor", smarts= "[C:2]=[C:3][#6,#7,#8,#15,#16:4]")
        self.alt_reactant2 = ReactionComponent(component="michael_acceptor", smarts= "[N:2]=[C:3][#6:4]")
        self.radical1 = ReactionComponent(component="radical-CCO",  parameters=[0,2])
        self.radical2 = ReactionComponent(component="radical-C", parameters=[0,2])
        self.m_adduct = ReactionComponent(component="adduct", smarts= "[C:1][C:2][C-:3][#6,#7,#8,#15,#16:4]", parameters=[0,2])
        self.alt_m_adduct = ReactionComponent(component="adduct", smarts = "[C:1][C:3]([#6:4])[N-:2]", parameters=[0,2])
        self.reaction_SMARTS = f"{self.reactant1.smarts}.{self.reactant2.smarts}>>{self.m_adduct.smarts}"
        self.alt_SMARTS = f"{self.reactant1.smarts}.{self.alt_reactant2.smarts}>>{self.alt_m_adduct.smarts}"
        
        #### DEBUGGING STUFF BY BOB
        self.logger = logger
        self.debug_save_location = debug_save_location
        ### END OF DEBUGGING

        self.assign_roles()

    def assign_roles(self) -> None:
        """
        This function contains the logic to assign the various roles specific to the reaction.
        It populates also the radicals attribute by calling the specific functions.
        """
        
        ### reactant1 ###
        reactant1_smiles = self.rsmi.split('|')[0]
        self.reactant1.smiles = reactant1_smiles
        
        reactant1_rdmol = Chem.MolFromSmiles(reactant1_smiles)
        reactant1_mol_obj = Loader.molecule_from_rdkit(reactant1_rdmol)
        reactant1_mol_obj.name = self.reactant1.component + "_" + self.name
        self.reactant1.structure.append(reactant1_mol_obj)
        self.reactant1.parameters = get_chg_and_mult(reactant1_rdmol)
        
        ### CHECK if we should use alternative ###
        reactant2_smiles = self.rsmi.split('|')[1]
        reactant2_mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(reactant2_smiles))
        check_smarts = Chem.MolFromSmarts(self.reactant2.smarts)
        reactant2_rdkit_mol = Chem.MolFromSmiles(reactant2_smiles)
        if not reactant2_rdkit_mol.HasSubstructMatch(check_smarts):
            self.reactant2 = self.alt_reactant2
            self.reaction_SMARTS = self.alt_SMARTS
        
        ### reactant2 ###
        reactant2_smiles = self.rsmi.split('|')[1]
        self.reactant2.smiles = reactant2_smiles
        
        reactant2_mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(reactant2_smiles))
        reactant2_mol_obj.name = self.reactant2.component + "_" + self.name
        self.reactant2.structure.append(reactant2_mol_obj)
        self.reactant2.parameters = get_chg_and_mult(Chem.MolFromSmiles(reactant2_smiles))

        ### radical1 ###
        radical1_mol_obj = Loader.molecule_from_rdkit(self.decomposition(reactant1_rdmol, "[C:1][C:5](=[O:6])[O-:7]"))
        radical1_mol_obj.name = self.radical1.component
        self.radical1.structure.append(radical1_mol_obj)
        
        ### radical2 ###
        
        radical2_mol_obj = Loader.molecule_from_rdkit(self.decomposition(reactant1_rdmol, "[C:1]"))
        radical2_mol_obj.name = self.radical2.component
        self.radical2.structure.append(radical2_mol_obj)

        ### m_adduct ###
        smarts_object = AllChem.ReactionFromSmarts(self.reaction_SMARTS)
        reactant1_rdmol = Chem.MolFromSmiles(self.reactant1.smiles)
        reactant2_rdmol = Chem.MolFromSmiles(self.reactant2.smiles)
        ps_i = smarts_object.RunReactants((reactant1_rdmol, reactant2_rdmol))
        
        # BOBS DEBUGGING STUFF HERE
        def save_mol_image(mol_obj: Chem.rdchem.Mol, mol_name: str, filename: str) -> None:
            try:
                self.logger.info(f"Trying to save mol_obj: {mol_name} as {filename}.png")
                debug_folder = Path(self.debug_save_location.value)
                filepath = Path(debug_folder / f"{filename}.png")
                Draw.MolToFile(mol=mol_obj, legend=mol_name, filename=filepath, imageType="png")
            except Exception as e:
                self.logger.info(f"Error while trying to save mol image for: {mol_name}!")

        def save_smiles_file(smiles: str, filename: str) -> None:
            self.logger.info(f"Debug save location: {self.debug_save_location}\nvalue:{self.debug_save_location.value}")
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.txt")
            with open(filepath, "a") as file:
                file.write(smiles)

        def save_to_xyz(structure_obj: Structure, filename: str) -> None:
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.xyz")
            structure_obj.write_xyz(path=filepath)

        def save_mol_obj(mol_obj: Structure, filename: str) -> None:
            debug_folder = Path(self.debug_save_location.value)
            filepath = Path(debug_folder / f"{filename}.pk")
            with open(filepath, "wb") as file:
                pickle.dump(mol_obj, file)

        self.logger.info(f"DEBUGGING:\nreactant 1: {reactant1_smiles}")
        self.logger.info(f"DEBUGGING:\nreactant 2: {reactant2_smiles}")

        # debugging: saving the mol IMAGE files
        save_mol_image(reactant1_rdmol, reactant1_mol_obj.name, "reactant_1")
        save_mol_image(reactant2_rdmol, reactant2_mol_obj.name, "reactant_2")
        #save_mol_image(radical1_mol_obj, radical1_mol_obj.name, "radical_1")
        #save_mol_image(radical2_mol_obj, radical2_mol_obj.name, "radical_2")

        # debugging: saving smiles for those that we have (for some reason the radicals objects dont want
        # to be turned into smiles, rdkit crashes every time I try to do anything with it)
        save_smiles_file(reactant1_smiles, "reactant1_smiles")
        save_smiles_file(reactant2_smiles, "reactant2_smiles")
        save_smiles_file(self.m_adduct.smiles, "m_adduct_smiles")
        # debugging: saving mol objects
        save_mol_obj(reactant1_rdmol, "m_adduct_smiles")
        save_mol_obj(reactant2_rdmol, "m_adduct_smiles")
        save_mol_obj(radical1_mol_obj, "radical1")
        save_mol_obj(radical2_mol_obj, "radical2")
        save_to_xyz(radical1_mol_obj, "radical1")
        save_to_xyz(radical2_mol_obj, "radical2")

        for prod_idx in range(len(ps_i)):
            product = ps_i[prod_idx][0]
            self.logger.info(f"SMILES of m_adduct: {self.m_adduct.smiles}. Now trying to save mol image...")
            self.m_adduct.smiles = Chem.MolToSmiles(product)
            save_mol_image(product, self.m_adduct.smiles, f"adduct")
        m_adduct_mol_obj = Loader.molecule_from_rdkit(Chem.MolFromSmiles(self.m_adduct.smiles))
        m_adduct_mol_obj.name = self.m_adduct.component
        self.m_adduct.structure.append(m_adduct_mol_obj)
        
    
    def decomposition(self, mol: Chem.rdchem.Mol, SMARTS_decomp: str) -> Chem.rdchem.Mol:
        """
        The functions takes a SMARTS pattern and an RDkit molecule as inputs and performs 
        a decompositon of the molecule affording the radical corresponding to the 
        transformation represented in the SMARTS pattern.

        Parameters
        ----------
        mol
            RDkit molecule object for the reactant.
        SMARTS_decomp
            SMARTS string of the substructure to remove.

        Returns
        -------
        Chem.rdchem.Mol
            Molecule object of the radical product.
        """

        product_decomp = copy.deepcopy(mol)
        smarts_reactant = "[C:1][C:5](=[O:6])[O:7][N]1[C](=[O])[c]2[c][c][c][c][c]2[C]1=[O]"
        reaction_SMARTS_decomp = f"{smarts_reactant}>>{SMARTS_decomp}"
        smirks_object = AllChem.ReactionFromSmarts(reaction_SMARTS_decomp)

        smirks_object.RunReactantInPlace(product_decomp)


        return product_decomp

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for
        each species in the reaction mechanism that has been populated

        Returns
        -------
        list[ReactionComponent]
            List of ReactionComponents objects created successfully.
        """

        available = []
        std_reaction_components = [
            self.solvent,
            self.reactant1,
            self.reactant2,
            self.radical1,
            self.radical2,
            self.m_adduct
            ]

        for f in std_reaction_components:
            if f.structure:
                available.append(f)

        return available