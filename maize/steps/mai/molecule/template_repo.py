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
from typing import cast, Literal, Tuple, Dict, Callable, Optional
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
    Bond,
    Atom,
    EntryCoord,
    Loader
)
from rdkit.Chem import rdChemReactions

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


def check_distance(collection1: NDArray[np.float32], collection2: NDArray[np.float32], threshold: float=0.7) -> bool:
    for point1 in collection1:
        for point2 in collection2:
            distance = np.sqrt(np.sum((point1 - point2)**2))
            if distance < threshold:
                return True
    return False

def find_perpendicular_axes(initial_axis : NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]: 
    # Normalize the initial axis
    initial_axis = initial_axis / np.linalg.norm(initial_axis)
    
    # Find a non-parallel axis
    non_parallel_axis = np.array([1, 0, 0]) if initial_axis[0] != 1 else np.array([0, 1, 0])
    
    # Find the first perpendicular axis using cross product
    perp_axis1 = np.cross(initial_axis, non_parallel_axis)
    perp_axis1 = perp_axis1 / np.linalg.norm(perp_axis1)  # Normalize the axis

    # Find the second perpendicular axis, which is perpendicular to both the initial axis and the first perpendicular axis
    perp_axis2 = np.cross(initial_axis, perp_axis1)
    perp_axis2 = perp_axis2 / np.linalg.norm(perp_axis2)  # Normalize the axis
    
    return perp_axis1, perp_axis2


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
    unsc_close_cent, unsc_far_cent = find_closest_centroid(subst_array, subst_attach_point)
    subst_o = np.array([at - subst_attach_point for at in subst_array], dtype=np.float32)
    # translate substituent coordinates, by putting the attach point at the origin
    or_att_pt = unsc_close_cent - subst_attach_point
    scaled_long_cent = unsc_far_cent - subst_attach_point
    close_subst_centroid = np.array(or_att_pt, dtype=np.float32)
    far_subst_centroid = np.array(scaled_long_cent)
    # get coordinates, labels and attach point from template molecule object
    # template list of coordinates
    template_attach_atom = template_dict[
        index
    ]  # identify the attach atom (label + coordinates) on the template based on the numerical index

    temp_attach_point = np.array(
        template_attach_atom.coords, dtype=np.float32
    )  # coordinates of the attach atom
    keys_temp = [at.element for at in template_dict if at.element]  # store atom labels
    scaled_template_coordinates: list[list[float]] = [
        list(np.array(at.coords, dtype=np.float32) - temp_attach_point) for at in template_dict
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

    new_temp_array = [
        np.array(coord.coords, dtype=np.float32) for coord in new_temp
    ]

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
    
   
    r1 = R.from_rotvec((np.pi - alfa) * norm_cross)  # rotation vector
    rot1 = np.array(r1.apply(subst_o), dtype=np.float32)  # rotated coordinates
    c_r_cent = find_closest_centroid(rot1, rot1[attach_index])[0] 
    

    # check the angle between the two centroids, if not zero,
    # rotates for the complementary angle and recalculate centroid
    if not np.allclose(r, c_r_cent, atol=1e-01):
        r_neg = R.from_rotvec((alfa - np.pi) * norm_cross)
        rot1 = np.array(r_neg.apply(subst_o), dtype=np.float32)
        c_r_cent = find_closest_centroid(rot1, rot1[attach_index])[0]

   
    #### Second rotation:
    # Rotate around the new formed axis to find the orientation that minimizes
    # the sterical hindrance. Assess different rotation angles and calculate
    # average distance for each atom from the centroid. The angle that maximizes
    # interatomic distance is assumed to be forming the best conformation.

    rotamer = np.array([])
    distance = 0.0
    rot2_angles = np.arange(0, 3.14, 0.02, dtype=np.float32)  # range of angles of rotation

    # go through a range of angles of rotation to check which is the optimal orientation.
    for beta in rot2_angles:

        ax_rot2 = (
            np.array(metal_cent.coords, dtype=np.float32) - far_subst_centroid
        )  # axis of rotation: vector between centroid and metal center
        
        norm_ax_rot2 = np.array(
        ax_rot2 / np.sqrt(np.dot(ax_rot2, ax_rot2)), dtype=np.float32
    )
        
        r2 = R.from_rotvec(beta * norm_ax_rot2)  # second rotation vector
        rot2 = np.array(r2.apply(rot1), dtype=np.float32)  # rotated conformer

        # check average atoms distances
        fdist = [cast(float, np.linalg.norm(k - temp_centroid)) for k in rot2]
        avg_dist = sum(fdist) / len(fdist)

        
        if not check_distance(rot2, new_temp_array):
            # store the rotamer only if the average distance is higher than what already stored.
            if avg_dist > distance:
                distance = avg_dist
                rotamer = rot2

    if len(rotamer) == 0:
   
        ax_rot3 = find_perpendicular_axes(ax_rot2)[0]
        for gamma in rot2_angles:
            r3 = R.from_rotvec(gamma * ax_rot3)
            rot3 = np.array(r3.apply(rot1), dtype=np.float32)  # rotated conformer

        # check average atoms distances
            fdist = [cast(float, np.linalg.norm(k - temp_centroid)) for k in rot3]
            avg_dist = sum(fdist) / len(fdist)

            if not check_distance(rot3, new_temp_array):
            # store the rotamer only if the average distance is higher than what already stored.
                if avg_dist > distance:
                    distance = avg_dist
                    rotamer = rot3
    
   
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

    
    # we need to keep track of the number of ligands (both coord and cov) that will be attached. 
    # This number will be used to readjust the index of atoms in the final molecule and store the correct
    # atomic indexes.

    coord_corrector = 0
    cov_corrector = 0

    if all(isinstance(x, int) for x in index_coord):
        coord_corrector = len(index_coord)
    if all(isinstance(y, int) for y in index_cov):
        cov_corrector = len(index_cov)
    
    index_corrector = coord_corrector + cov_corrector # tot number of indexes in the templates
    
    template.connectivity_by_distance()
    intermediate_adduct = template.mol_dict
    intermediate_connectivity = template.connectivity

    # get connectivity matrix. Numbers have to be scaled by removing the number of atoms that are
    # used as placeholders for interactions in the templates. These are always the first ones in 
    # the template therefore once the index is subtracted with the index corrector they will become
    # either negative or zero. Thus we can use a simple logic to select the ones we are intrested in.

    intermediate_cmatrix = [(tx - index_corrector + 1, ty - index_corrector + 1, tz) for tx, ty, tz in template.c_matrix if (tx - index_corrector + 1) > 0 and (ty - index_corrector + 1 > 0)]

    # store the index of the atom in the template to fix for future calculations
    temp_fix_indexes = len(intermediate_adduct) - len(index_coord)
    
    fix_index = list(range(temp_fix_indexes))  # list of atom index to fix. Add the atoms that are part of the template.

    # Ligand SMILES is converted to a Structure object.
    lig_coord_rdmol = Chem.MolFromSmiles(ligand_coord)
    mol_lig_coord = Loader.molecule_from_rdkit(lig_coord_rdmol)
    mol_lig_coord.connectivity_by_distance()
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

    clean_ligand_cov.connectivity_by_distance()
    clean_lig_cov_dict = clean_ligand_cov.mol_dict
    adjust_att_pt = clean_lig_cov_dict[adjusted_index]
    
    # Check for covalent interaction in the template. Attach the ligand covalently if any.
    m = 1
    reag2_idx= None
    intermediate_adduct = template.mol_dict
    for index in index_cov:
        if type(index) is int:
            cov_adj = len(index_cov) - m 
            istance_n, new_template, cov_markers = orient_substituent(
                template.mol_dict, clean_lig_cov_dict, adjust_att_pt, index
            )
            
            intermediate_adduct = new_template
            met_idx = cov_markers[1]
            
            
            adjust_cov_idx = 0
            if isinstance(index_coord[0], int):
                adjust_cov_idx = len(index_coord)

            for idx in index_coord:
                if isinstance(idx, int):
                    idx = idx - 1 

            cov_idx = cov_markers[0]
            att_nr = None
            met_nr = None
           
            # readjust indexes of coordination interaction point
            # after adding ligand and removing the attach point.
            adjust = len(intermediate_adduct) - adjust_cov_idx + 1
            # append the coordinates of the ligand to the final intermediate structure
            for q in istance_n:
                intermediate_adduct.append(q)

            # adjust connectivity indexes by adding the length of the existing adduct 
            # and subtracting the indices of the attach point still not removed. (Plus one
            # for moving from 0 to 1 index.) 
            adjusted_cm = [(cmx + adjust, cmy + adjust, cmz) for cmx, cmy, cmz in clean_ligand_cov.c_matrix]
            for cm in adjusted_cm: 
                intermediate_cmatrix.append(cm)
            
            # append the bonds of the ligand to the final intermediate connectivity list
            for legame in clean_ligand_cov.connectivity:
                intermediate_connectivity.append(legame)

            cov_bond = Bond(cov_idx.to_Atom(), met_idx.to_Atom(), 1)
            intermediate_connectivity.append(cov_bond)
            
            met_index_ref = 0
            for indice, atm in enumerate(intermediate_adduct):
               
                # get index for metal center
                if atm.coords == met_idx.coords and atm.element == met_idx.element:
                    
                    if indice not in fix_index:
                        fix_index.append(indice)

                    if indice + 1 > len(index_cov):
                        met_nr = indice - cov_adj - adjust_cov_idx + 1
                        met_index_ref = indice
                         
                        
                    else:
                        met_nr = indice + 1
                        met_index_ref = indice

                # get index for attach point on ligand
                elif atm.coords == cov_idx.coords and atm.element == cov_idx.element:
                    reag2_idx = indice - adjust_cov_idx
                    fix_index.append(reag2_idx)
                  
                    if met_index_ref + 1 > len(index_cov):
                        att_nr = indice - cov_adj - adjust_cov_idx + 1
                        
                    else:
                        att_nr = indice + 1
                 
                    m += 1 
            
            if att_nr and met_nr:
                intermediate_cmatrix.append((met_nr, att_nr, 1.0))
                
                

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
            att_number = None
            metal_cent = markers[1]
            met_number = None
    
            adjsutement = len(intermediate_adduct) - idx_adjust + 1
            # append the ligand atom coordinates to the final intermediate structure.
            for q in istance_n:
                intermediate_adduct.append(q)


            for i, coord in enumerate(intermediate_adduct):
                if coord.coords == metal_cent.coords and coord.element == metal_cent.element:
                    if i not in fix_index:
                        fix_index.append(i)
                    
                    if i + 1 > idx_adjust:
                        met_number = i - idx_adjust + 1    
                    else:
                        met_number = i + 1
 
                elif coord.coords == att_pt.coords and coord.element == att_pt.element:
                    att_number = i - idx_adjust + 1 
                    fix_index.append(att_number)  # append the adjusted indexes to the list of atom index to fix.
                    n += 1
                
            adjusted_zm = [(zmx + adjsutement, zmy + adjsutement, zmz) for zmx, zmy, zmz in mol_lig_coord.c_matrix]
            for zm in adjusted_zm:    
                intermediate_cmatrix.append(zm)
            
            for lien in mol_lig_coord.connectivity:
                intermediate_connectivity.append(lien)
            # append newly formed bond as well
            coord_bond = Bond(att_pt.to_Atom(), metal_cent.to_Atom(), 1)
            intermediate_connectivity.append(coord_bond)
  
            if met_number and att_number:
                intermediate_cmatrix.append((met_number, att_number, 1.0))
    # create compchem_utils molecule object of the output adduct structure
    final_structure = []
    for number, line in enumerate(intermediate_adduct):
        atom = Atom(line.element, [line.coords[0], line.coords[1], line.coords[2]])     
        atom.number = number
        final_structure.append(atom)

    final_mol = Structure(final_structure)
    final_mol.c_matrix = intermediate_cmatrix
    final_mol.name = text

    # adjust the atomic index from 0-indexed to 1-indexed. For compatibility with comp chem software
    fix_index_sdf = [a + 1 for a in fix_index]
    
    
    if reag2_idx:
        reag2_idx =  reag2_idx + 1

    return final_mol, fix_index_sdf, reag2_idx



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
    coord_idx: list[int] | None = None
    ar_carbon: int | None = None 
    alk_carbon: int | None = None
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
            connectivity_info = mk_mol.c_matrix
            
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
                isomer_mol.set_tag("connectivity", connectivity_info)
            else:
                connectivity_info_plus1 = [(mx + 1, my + 1, mz) for mx, my, mz in connectivity_info]
                isomer_mol.set_tag("connectivity", connectivity_info_plus1)
            
            if self.coord_idx:
                n_coord = self.coord_idx[i]
                isomer_mol.set_tag("n_coord", n_coord)

            if self.ar_carbon:
                isomer_mol.set_tag("ar_carbon", self.ar_carbon)
            
            if self.alk_carbon:
                isomer_mol.set_tag("alk_carbon", self.alk_carbon)
            

            list_isomers.append(isomer_mol)

        return list_isomers
        
@dataclass(frozen=True)
class DAG_transform_node:

    """
    Class type for storing a transformation on a ReactionComponent.
    It will take in a dictionary <component name:str, component:ReactionComponent>
    this transformation will perform a boolean check on the SMILES strings of the reactioncomponents.
    If True: perform an rdkit reaction with a given SMARTS formula, creating new component(s) and return True.
    If False: return False and the dict as is.

    ---
    Attributes:
    required_component_names : list<str>
        lists the names of the required ReactionComponents as strings. These names are stored in the 'component'
        field of the ReactionComponent object as well.
    transformation_boolean_checks : dict<str, str>
        maps component_name to SMARTS strings to check if component has expected form.
        if ALL match we return True, otherwise we dont perform transformations and return False
    transformation_inputs : dict<str, str>
        dict of 'component name' -> 'component SMARTS' strings for the reaction,
        this is a dict to make sure the right input SMARTS are lined up to the right component SMILES
    transformation_outputs : list[Tuple[str,str]]
        output SMARTS of the transformation. Will be concatenated to the transformation_inputs with '>>'
    """

    required_components: list[str] = field(default_factory=list, hash=False)
    transformation_checks: Dict[str,str] = field(default_factory=dict, hash=False)
    transformation_inputs: Dict[str,str] = field(default_factory=dict, hash=False)
    transformation_outputs: Dict[str,str] = field(default_factory=dict, hash=False)
    name: Optional[str] = "Default DAG transformation name"

    def __call__(self, available_components : Dict[str, list[ReactionComponent]]) -> tuple[Dict[str, list[ReactionComponent]], bool]:
        # To do: check if the required components line up with transformation_check and transformation_inputs

        # check if all the selected_components are present in the components dict
        for required_component in self.required_components:
            if available_components.get(required_component, []) == []:
                # return the dict as is and False as well.
                return available_components, False

        # check if all the transformation_checks SMARTS actually match so if we 
        for component_name, component_check_SMARTS in self.transformation_checks.items():
            # create a mol object for the check and check hassubstructmatch
            mol_check_obj = Chem.MolFromSmarts(component_check_SMARTS)
            # retrieve the reactioncomponent object from the available_components
            relevant_component_list = available_components[component_name]
            # create a mol object from the reactioncomponent to check it with
            for relevant_component in relevant_component_list:
                # compare it with the smiles in the reactioncomponent
                component_SMILES = relevant_component.smiles
                component_rdmol = Chem.AddHs(Chem.MolFromSmiles(component_SMILES))
                if not component_rdmol.HasSubstructMatch(mol_check_obj):
                    return available_components, False

        # start building the reaction SMARTS input string
        reaction_SMARTS_list = []
        reactioncomponent_inputs = []
        for component_name, component_SMARTS in self.transformation_inputs.items():
            # To do: change to apply reactions to ALL available components
            # ... this might get more complicated than it seems, because you would have to run it on all combinations of
            # ... ReactionComponents in all lists that go into the reaction
            relevant_component = available_components[component_name][0]
            rdmol_obj = Chem.AddHs(Chem.MolFromSmiles(relevant_component.smiles))
            reactioncomponent_inputs.append(rdmol_obj)
            reaction_SMARTS_list.append(component_SMARTS)

        # start building the reaction SMARTS output string
        reaction_SMARTS = ".".join(reaction_SMARTS_list)
        reactioncomponent_outputs = []
        output_component_names = []
        for output_component_name, output_SMARTS in self.transformation_outputs.items():
            reactioncomponent_outputs.append(output_SMARTS)
            output_component_names.append(output_component_name)
        reaction_SMARTS += ">>" + (".".join(reactioncomponent_outputs))

        # now execute the reaction and return all the new components
        reaction = AllChem.ReactionFromSmarts(reaction_SMARTS)
        products = reaction.RunReactants(tuple(reactioncomponent_inputs))
        for product_set in products:
            for i, product in enumerate(product_set):
                product_without_hs = Chem.RemoveHs(product)
                component_name = output_component_names[i]
                structure_obj = Loader.molecule_from_rdkit(product_without_hs)
                structure_obj.name = output_component_names[i]
                current_reactioncomponent = ReactionComponent(
                    component=component_name,
                    smiles=Chem.MolToSmiles(product_without_hs),
                    parameters=get_chg_and_mult(product_without_hs),
                    structure=[structure_obj])
                reactioncomponent_list = available_components.get(component_name, [])
                reactioncomponent_list.append(current_reactioncomponent)
                available_components[component_name] = reactioncomponent_list
        return available_components, True

@dataclass(frozen=True)
class convert_anion_to_radical_node:
    """
    Class for converting anionic atoms to radical species in molecular components.

    This class is designed to be used as a callable object within a molecular transformation pipeline.
    It targets specific components and converts their anionic atoms (with a -1 formal charge) into
    radical species (with 0 formal charge and 1 radical electron).

    ---
    Attributes:
    components_to_radicalize : list[str]
        A list of component names to be processed for anion-to-radical conversion.

    Methods:
    __call__(available_components)
        Executes the anion-to-radical conversion on specified components.

        Parameters:
        available_components : Dict[str, list[ReactionComponent]]
            A dictionary of component names mapped to lists of RDKit Mol objects.

        Returns:
        tuple[Dict[str, list[ReactionComponent]], bool]
            A tuple containing the modified components dictionary and a boolean (always True).

    Notes:
    - The conversion process finds the first anionic atom in each molecule of the specified components.
    - The anionic atom's formal charge is set to 0, and its number of radical electrons is set to 1.
    - After conversion, the molecule is sanitized and explicit hydrogens are added.
    """
    affected_component_name: str
    new_component_name: str
    name: Optional[str] = "Default DAG transformation name"

    def __call__(self, available_components : Dict[str, list[ReactionComponent]]) -> tuple[Dict[str, list[ReactionComponent]], bool]:
        def add_radical_electron(component : ReactionComponent, new_component_name : str) -> Tuple[Dict[str, list[ReactionComponent]], bool]:
            # replace the negative charge on 1 atom with an electron, then set the charge to 0.
            # we do this because we're trying to replace the negative charge with the radical (unpaired electron).
            # this would have the same effect in principle (sort of)
            rdmol = Chem.MolFromSmiles(component.smiles)
            for atom in rdmol.GetAtoms():
                if atom.GetFormalCharge() == -1:
                    idx = atom.GetIdx()
                    atom.SetFormalCharge(0)
                    atom.SetNumRadicalElectrons(1)
                    break
            new_structure = Loader.molecule_from_rdkit(rdmol)
            new_structure.name = new_component_name
            new_component = ReactionComponent(
                component = new_component_name,
                smiles = Chem.MolToSmiles(rdmol),
                structure = [new_structure],
                parameters = get_chg_and_mult(rdmol))
            return new_component
        if not self.affected_component_name in available_components:
            return available_components, False
        available_components[self.new_component_name] = [add_radical_electron(component, self.new_component_name) for component in available_components[self.affected_component_name]]
        return available_components, True

@dataclass(frozen=True)
class add_structureless_single_electron:
    """
    Class for adding a single electron to a components without changing the structure, only the charge and multiplicity are affected.
    The electron addition lowers the formal charge by 1 and adjusts the multiplicity based on the assumption that unpaired electrons 
    will pair up.

    ---
    Attributes:
    affected_component_name : str
        The name of the component to which a single electron will be added.
    new_component_name : str
        The name assigned to the new component after the transformation.

    Methods:
    __call__(available_components)
        Executes the electron addition on the specified component.

        Parameters:
        available_components : Dict[str, list[ReactionComponent]]
            A dictionary that contains all the current components mapping component names to lists of ReactionComponent objects.

        Returns:
        tuple[Dict[str, list[ReactionComponent]], bool]
            A tuple with the updated components dictionary and a boolean (always True if the component exists).
            
    Notes:
    - This transformation assumes that any unpaired electrons present will pair up when the electron is added.
    - The charge is reduced by 1, and the multiplicity is adjusted accordingly.
    """

    affected_component_name: str
    new_component_name: str
    name: Optional[str] = "Default DAG transformation name"

    def __call__(self, available_components : Dict[str, list[ReactionComponent]]) -> tuple[Dict[str, list[ReactionComponent]], bool]:
        def add_single_electron(current_reactioncomponent : ReactionComponent, new_reactioncomponent_name: str) -> ReactionComponent:
            # lower the charge by 1
            # change the multiplicity as well, BY ASSUMING THAT ANY UNPAIRED ELECTRONS WILL PAIR UP!!!
            current_charge, current_multiplicity = current_reactioncomponent.parameters
            new_charge = current_charge - 1
            new_multiplicity = (current_multiplicity % 2) + 1
            new_structure_list = copy.deepcopy(current_reactioncomponent.structure)
            for structure in new_structure_list:
                structure.name = new_reactioncomponent_name
            return ReactionComponent(
                component = new_reactioncomponent_name,
                smiles = current_reactioncomponent.smiles,
                structure = new_structure_list,
                parameters = [new_charge, new_multiplicity]
            ) 
        if not self.affected_component_name in available_components:
            return available_components, False
        available_components[self.new_component_name] = [add_single_electron(component, self.new_component_name) for component in available_components[self.affected_component_name]]
        return available_components, True

@dataclass
class TRANSFORMATION_DAG:
    """
    Class for storing and executing a Directed Acyclic Graph (DAG) of SMILES transformations.

    This class represents a DAG where each node is a transformation function that operates on
    SMILES (Simplified Molecular Input Line Entry System) representations of molecules. The DAG
    structure allows for conditional branching based on the results of each transformation.

    ---
    Attributes:
    dag : Dict[Callable, Dict[bool, Callable]]
        A dictionary representing the DAG structure. Each key is a transformation function,
        and its value is another dictionary with boolean keys mapping to the next transformation
        function to be called based on the result of the current transformation.

    Methods:
    add_transformation_node(transformation_node, positive_node, negative_node)
        Adds a new transformation node to the DAG with specified positive and negative branches.

    run(start_transformation_node, begin_components)
        Executes the DAG starting from a specified transformation node with given initial components.
    """
    dag : Dict[Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]], Dict[bool, Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]]]]]] = field(hash=False)
    start_node = Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]]]]

    def __init__(self) -> None:
        self.dag: Dict[Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]], Dict[bool, Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]]]] = {}

    def designate_start_node(self, start_node : Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]]) -> None:
        self.start_node = start_node

    def add_transformation_node(self,
        transformation_node : Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]],
        positive_node : Optional[Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]]] = None,
        negative_node : Optional[Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]]] = None) -> None:
        node_dict = {}
        if positive_node:
            node_dict[True] = positive_node
        if negative_node:
            node_dict[False] = negative_node
        self.dag[transformation_node] = node_dict
    
    def run(self, components : Dict[str, list[ReactionComponent]]) -> Dict[str, list[ReactionComponent]]:
        if not self.start_node:
            raise Exception("Start node not set in instance of DAG! Don't know where to start! Set this via the designate_start_node function.")
        current_transformation_node = self.start_node
        while current_transformation_node:
            components, result = current_transformation_node(components)
            next_transformation_node = self.dag.get(current_transformation_node, {}).get(result, None)
            current_transformation_node = next_transformation_node
        return components

# function for turning a dict of components into a list of ReactionComponents
def prepare_initial_components(smiles_list : list[str], component_name_list : list[str]) -> Dict[str, list[ReactionComponent]]:
    initial_components = {}
    for i, smiles in enumerate(smiles_list):
        name = component_name_list[i]
        rdmol = Chem.MolFromSmiles(smiles)
        rdmol_extra_hs = Chem.AddHs(rdmol)
        structure_obj = Loader.molecule_from_rdkit(rdmol)
        structure_obj.name = name
        parameters = get_chg_and_mult(rdmol)
        initial_components[name] = [ReactionComponent(component=name,smiles=Chem.MolToSmiles(rdmol_extra_hs),structure=[structure_obj],parameters=parameters)]
    return initial_components


# here we actually remove all the extra hydrogens
def extract_and_convert_components(available_components : Dict[str, list[ReactionComponent]], to_ignore: list[str] = []) -> list[ReactionComponent]:
    components_to_return = []
    for component_name, component_list in available_components.items():
        if component_name in to_ignore:
            continue
        for component in component_list:
            components_to_return.append(component)
    return components_to_return


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
        int: dict[str, tuple[str, list[int], list[str] | list[int]]], geom: str
    ) -> tuple[Structure, list[int], list[str] | list[int]]:
        """
        This function takes as input an intermediate and a type of geometry and
        returns the respective template as Structure object and the associated
        indexes for covalent and coordinate interactions.

        Parameters
        ----------
        int
            numerical index of the template to retrieve
        geom
            geometry associated with the structure

        Returns
        -------
        tuple[Structure, list[int], list[int] | None]
            Structure object of the template,
            List of indexes corresponding to coordination interactions,
            List of indexes corresponding to covalent interactions
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
            test_iter, atms_to_fix, carbon = make_intermediate(
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

    def __init__(self, id: str, rsmi: str, roles: str, logger: logging.Logger, intermediate_save_location: FileParameter[Path] = None) -> None:
        """
        The constructor defines the roles for the reaction. These are for
        now specifically tailored for this particular reaction mechanism.
        In the future we shall evaluate if it is possible to generalize
        most of it for different reaction. Also check if the class format
        is necessary, should consider something more versatile such as
        JSON objects.

        """
        self.name = id
        self.logger: logging.Logger = logger
        self.intermediate_save_location: FileParameter[Path] = intermediate_save_location
        self.rsmi = rsmi
        self.others = roles
        self.solvent = ReactionComponent(component="solvent")
        self.reactant1 = ReactionComponent(component="reagent1")
        self.reactant2 = ReactionComponent(component="reagent2", smarts="[C:1]Br")
        self.bromide = ReactionComponent(component="Bromide", smiles="[Br-]")
        self.debromo_alk = ReactionComponent(component="Debrominated_alkyl")
        self.c_radical = ReactionComponent(component="alkyl_radical", parameters=[0,2])
        self.c_anion = ReactionComponent(component="alkyl_anion", parameters=[-1,1])
        self.c_cation = ReactionComponent(component="alkyl_cation", parameters=[1,1])
        self.amine = ReactionComponent(component="amine")
        self.Ni0_inter = ReactionComponent(component="Ni0_intermediate1")
        self.Ni1_inter = ReactionComponent(component="Ni1_intermediate2", parameters=[0, 2])
        self.Ni2_inter = ReactionComponent(component="Ni2_intermediate3")
        self.Ni3_inter = ReactionComponent(component="Ni3_intermediate4", parameters=[0, 2], ar_carbon=3)

        self.int_geometries = ["TrigonalBipyramidal"]
        self.intermediates_temp: dict[
            str, dict[str, tuple[str, list[int] | list[str], list[str] | list[int]]]
        ] = {
            "Ni0": {
                "TrigonalPlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/Ni0_trpl_CC_856.sdf",
                    [2, 1, 0],
                    ["NA"],
                ),
                "Tetrahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/Ni0_tetra_CC_856.sdf",
                    [3, 2, 1, 0],
                    ["NA"],
                ),
                "SquarePlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/Ni0_sqpl_CC_856.sdf",
                    [3, 2, 1, 0],
                    ["NA"]
                ),
                "TrigonalBipyramidal": (
                    f"{Path(__file__).parent}/templates/CC_templ/Ni0_trbpyr_CC_856.sdf",
                    [2, 1, 0],
                    ["NA"]
                ),
                "Octahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/Ni0_oct_CC_856.sdf",
                    [3, 2, 1, 0],
                    ["NA"]
                )
            },
            "Ni1": {
                "TrigonalPlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_trpl_CC_856.sdf",
                    [1, 0],
                    ["NA"],
                ),
                "Tetrahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_tetra_CC_856.sdf",
                    [2, 1, 0],
                    ["NA"],
                ),
                "SquarePlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_sqpl_CC_856.sdf",
                    [2, 1, 0],
                    ["NA"],
                ),
                "TrigonalBipyramidal": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_trbpyr_CC_856.sdf",
                    [1, 0],
                    ["NA"]
                ),
                "Octahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_oct_CC_856.sdf",
                    [2, 1, 0],
                    ["NA"]
                )
            },
            "Ni2": {
                "TrigonalPlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_trpl_CC_856.sdf",
                    [0],
                    ["NA"],
                ),
                "Tetrahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_tetra_CC_856.sdf",
                    [1, 0],
                    ["NA"],
                ),
                "SquarePlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_sqpl_CC_856.sdf",
                    [1, 0],
                    ["NA"],
                ),
                "TrigonalBipyramidal": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_trbpyr_CC_856.sdf",
                    [0],
                    ["NA"]
                ),
                "Octahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_oct_CC_856.sdf",
                    [1, 0],
                    ["NA"]
                )
            },
            "Ni3": {
                "TrigonalPlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_trpl_CC_856.sdf",
                    ["NA"],
                    [0],
                ),
                "Tetrahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_tetra_CC_856.sdf",
                    [0],
                    [1],
                ),
                "SquarePlanar": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_sqpl_CC_856.sdf",
                    [0],
                    [1],
                ),
                "TrigonalBipyramidal": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_trbpyr_CC_856.sdf",
                    ["NA"],
                    [0],
                ),
                "Octahedral": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_oct_CC_856.sdf",
                    [0],
                    [1],
                )
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
        self.reactant1.structure.append(mol_reac1)
        self.reactant1.parameters = get_chg_and_mult(reac1_rdmol)

        # reactant 2
        reac2_smiles = left_hand.split(".")[1]
        reac2_rdmol = Chem.MolFromSmiles(reac2_smiles)
        mol_reac2 = Loader.molecule_from_rdkit(reac2_rdmol)
        mol_reac2.name = self.reactant2.component
        self.reactant2.smiles = reac2_smiles
        self.reactant2.mw = mol_reac2.get_mw()
        self.reactant2.structure.append(mol_reac2)
        self.reactant2.parameters = get_chg_and_mult(reac2_rdmol)

        # amine
        amine_smiles = split_smiles_from_reaction(self.others)[1]
        amine_rdmol = Chem.MolFromSmiles(amine_smiles)
        mol_amine = Loader.molecule_from_rdkit(amine_rdmol)
        mol_amine.name = self.amine.component
        self.amine.smiles = amine_smiles
        self.amine.structure.append(mol_amine)
        self.amine.parameters = get_chg_and_mult(amine_rdmol)

        #bromide
        bromide_rdmol = Chem.MolFromSmiles(self.bromide.smiles)
        mol_br = Loader.molecule_from_rdkit(bromide_rdmol)
        mol_br.name = self.bromide.component
        self.bromide.structure.append(mol_br)
        self.bromide.parameters = get_chg_and_mult(bromide_rdmol)

        # decompositions
        
        debr_alk = Chem.AddHs(reac2_rdmol)
        edit = Chem.RWMol(debr_alk)
        for atom in edit.GetAtoms():
            if atom.GetAtomicNum() == 35:
                atom.SetAtomicNum(1)
        Chem.SanitizeMol(edit)
        Chem.RemoveStereochemistry(edit)
        debr_alk_rdmol = Chem.RemoveHs(edit)
        debr_alk_mol = Loader.molecule_from_rdkit(debr_alk_rdmol) # debrominated reactant
        debr_alk_mol.name = self.debromo_alk.component
        self.debromo_alk.structure.append(debr_alk_mol)
        self.debromo_alk.parameters = get_chg_and_mult(debr_alk_rdmol)


        alk_smarts = "[C:1]"
        alk_rdmol = Chem.MolFromSmiles(reac2_smiles)
        reaction_SMARTS = f"{self.reactant2.smarts}>>{alk_smarts}"
        smirks_object = AllChem.ReactionFromSmarts(reaction_SMARTS)
        smirks_object.RunReactantInPlace(alk_rdmol)
    
        alkrad_mol = Loader.molecule_from_rdkit(alk_rdmol) # alkyl radical
        alkrad_mol.name = self.c_radical.component
        self.c_radical.structure.append(alkrad_mol)

        alkan_mol = Loader.molecule_from_rdkit(alk_rdmol)  # anion
        alkan_mol.name = self.c_anion.component
        self.c_anion.structure.append(alkan_mol)
 
        alkcat_mol = Loader.molecule_from_rdkit(alk_rdmol)  # cation
        alkcat_mol.name = self.c_cation.component
        self.c_cation.structure.append(alkcat_mol)


        # solvent
        solv_smiles = self.others.split(".")[-1]
        solv_rdmol = Chem.MolFromSmiles(solv_smiles)
        mol_solv = Loader.molecule_from_rdkit(solv_rdmol)
        mol_solv.name = self.solvent.component
        self.solvent.smiles = solv_smiles
        self.solvent.mw = mol_solv.get_mw()
        self.solvent.structure.append(mol_solv)
        self.solvent.parameters = get_chg_and_mult(solv_rdmol)


        # Ni0 inter
        # self.Ni0_inter.structure, self.Ni0_inter.constraints, self.Ni0_inter.coord_idx, self.Ni0_inter.alk_carbon  = self.create_intermediates("Ni0")
        
        # Ni1 inter
        # self.Ni1_inter.structure, self.Ni1_inter.constraints, self.Ni1_inter.coord_idx, self.Ni1_inter.alk_carbon = self.create_intermediates("Ni1")      

        # Ni2 inter
        # self.Ni2_inter.structure, self.Ni2_inter.constraints, self.Ni2_inter.coord_idx, self.Ni2_inter.alk_carbon = self.create_intermediates("Ni2")

        # Ni3 inter
        self.Ni3_inter.structure, self.Ni3_inter.constraints, self.Ni3_inter.coord_idx, self.Ni3_inter.alk_carbon = self.create_intermediates("Ni3")
       


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
    ) -> tuple[list[Structure], list[list[int]], list[int], int | None]:
        """
        This function creates the proper intermediate molecule
        object and the associated atom to be constrained.

        Parameters
        ----------
        intermediate_template
            The template structure for the intermediate
        
        Returns
        -------
        tuple[list[Structure], list[list[int]], list[int], int]
            list[Structure] contains Structure objects of intermediates
            list[list[int]] contains list of atom indexes for constraints
            list[int] contains list of coord indexes to store as tags
            int|None contains the index of the carbon atom to check for collapsed structures 

        """
        intermediates_list: list[Structure] = []
        constraints_list: list[list[int]] = []

        intermediates_list = []
        constraints_list = []
        coord_list = []

        for geom in self.int_geometries:
            temp_mol, coord_idxs, cov_idxs = NiCatCycle_CC.retrieve_template(
                self.intermediates_temp[intermediate_template], geom
            )
           
            test_iter, atms_to_fix, carbon = make_intermediate(
                temp_mol,
                coord_idxs,
                cov_idxs,
                self.solvent.smiles,
                self.reactant2.smiles,
                "{}_{}".format(intermediate_template, geom),
            )

            if "NA" in coord_idxs:
                coord_list.append(0)
            else:
                coord_list.append(len(coord_idxs))
            
            intermediates_list.append(test_iter)
            constraints_list.append(atms_to_fix)
            
        return intermediates_list, constraints_list, coord_list, carbon

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for each
        species in the reaction mechanism that has been populated

        """

        available: list[ReactionComponent] = []
        std_reaction_components = [
            self.reactant1,
            self.reactant2,
            self.Ni3_inter
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
        reactant1_rdmol = Chem.MolFromSmiles(self.reactant1.smiles) # maybe change this to reactant1_smiles
        reactant2_rdmol = Chem.MolFromSmiles(self.reactant2.smiles) # maybe change this as well
       
        ps_i = smarts_object.RunReactants((reactant1_rdmol, reactant2_rdmol))
        
        # BOBS DEBUGGING STUFF HERE
        def save_mol_image(mol_obj : Chem.rdchem.Mol, mol_name : str, filename : str) -> None:
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

        def save_mol_obj(mol_obj : Chem.rdchem.Mol, filename : str) -> None:
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


class ConditionalRadicalTransformations:

    def __init__(self, id: str, rsmi: str, logger: logging.Logger, intermediate_save_location: FileParameter[Path]) -> None:
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
        self.name: str = id
        self.rsmi: str = rsmi
        self.solvent = ReactionComponent(component="solvent")
        self.reaction_components: list[ReactionComponent] = [ReactionComponent]
        self.dag: TRANSFORMATION_DAG = TRANSFORMATION_DAG()
        self.logger: logging.Logger = logger
        self.intermediate_save_location: FileParameter[Path] = intermediate_save_location

        self.create_TRANSFORMATION_DAG()
        self.execute_TRANSFORMATION_DAG()

    def _find_smarts_in_smiles(self, smiles: str, smarts: str) -> bool:
        rdkit_mol = Chem.MolFromSmiles(smiles)
        checking_smarts = Chem.MolFromSmarts(smarts)
        if rdkit_mol.HasSubstructMatch(checking_smarts):
            return True
        return False

    def create_TRANSFORMATION_DAG(self) -> None:
        """
        This function creates a DAG of the transformation steps for the input SMILES
        """
        # define the SMARTS to be used
        self.logger.info("Running create_TRANSFORMATION_DAG")
        is_ester_SMARTS_input = "[C:1][C:5](=[O:6])[O:7][N]1[C](=[O])[c]2[c][c][c][c][c]2[C]1=[O]"
        is_acid_SMARTS_input = "[C:1][C:5](=[O:6])[O:7][H]"
        has_boron_SMARTS_input = "[C:1][B]([O])([O])"
        C_radical_pattern = "[C-:1]"
        CCO_radical_pattern = "[C:1][C:5](=[O:6])[-O:7]"
        michael_acceptor_input_A = "[C:2]=[C:3][#6,#7,#8,#15,#16:4]"
        michael_acceptor_input_B = "[N:2]=[C:3][#6,#7,#8,#15,#16:4]"
        adduct_A = "[C:1][C:2][C-:3][#6,#7,#8,#15,#16:4]"
        adduct_B_1 = "[C:1][C:3]([#6:4])[N-:2]"
        adduct_B_2 = "[C:1][N:2][C-:3][#6:4]"
        
        ### define the nodes for the ester path
        ester_to_radicals_node = DAG_transform_node(
            transformation_checks={'reactant': is_ester_SMARTS_input},
            transformation_inputs={'reactant': is_ester_SMARTS_input},
            transformation_outputs={"anion_radical_C": C_radical_pattern, "anion_radical_CCO": CCO_radical_pattern},
            name="ester_to_radicals_node",
            )
        
        ester_radical_and_MA_to_adduct_A = DAG_transform_node(
            transformation_checks={'michael_acceptor':michael_acceptor_input_A},
            transformation_inputs={'reactant': is_ester_SMARTS_input, 'michael_acceptor':michael_acceptor_input_A},
            transformation_outputs={'adduct': adduct_A},
            name="ester_radical_and_MA_to_adduct_A",
            )
        ester_radical_and_MA_to_adduct_B = DAG_transform_node(
            transformation_checks={'michael_acceptor':michael_acceptor_input_B},
            transformation_inputs={'reactant': is_ester_SMARTS_input, 'michael_acceptor':michael_acceptor_input_B},
            transformation_outputs={'adduct': adduct_B_1, 'adduct': adduct_B_2},
            name="ester_radical_and_MA_to_adduct_B",
            )

        ### define the nodes for the acid path
        acid_to_radicals_node = DAG_transform_node(
            transformation_checks={'reactant': is_acid_SMARTS_input},
            transformation_inputs={'reactant': is_acid_SMARTS_input},
            transformation_outputs={"anion_radical_C": C_radical_pattern, "anion_radical_CCO": CCO_radical_pattern},
            name="acid_to_radicals_node",
            )
        acid_radical_and_MA_to_adduct_A = DAG_transform_node(
            transformation_checks={'michael_acceptor':michael_acceptor_input_A},
            transformation_inputs={'reactant': is_acid_SMARTS_input, 'michael_acceptor':michael_acceptor_input_A},
            transformation_outputs={'adduct': adduct_A},
            name="acid_radical_and_MA_to_adduct_A",
            )
        acid_radical_and_MA_to_adduct_B = DAG_transform_node(
            transformation_checks={'michael_acceptor':michael_acceptor_input_B},
            transformation_inputs={'reactant': is_acid_SMARTS_input, 'michael_acceptor':michael_acceptor_input_B},
            transformation_outputs={'adduct': adduct_B_1, 'adduct': adduct_B_2},
            name="acid_radical_and_MA_to_adduct_B",
            )

        ### define the nodes common to all paths
        radicalize_radical_C = convert_anion_to_radical_node( # common to all paths
            affected_component_name="anion_radical_C",
            new_component_name="radical_C",
            name="radicalize_radical_C",
            )
        radicalize_radical_CCO = convert_anion_to_radical_node( # common to all paths
            affected_component_name="anion_radical_CCO",
            new_component_name="radical_CCO",
            name="radicalize_radical_CCO",
            )
        radicalize_adduct = convert_anion_to_radical_node( # common to all paths
            affected_component_name="adduct",
            new_component_name="radical_adduct",
            name="radicalize_adduct",
            )

        # put the nodes into the DAG structure
        self.dag.designate_start_node(ester_to_radicals_node)
        self.dag.add_transformation_node(ester_to_radicals_node, positive_node=ester_radical_and_MA_to_adduct_A, negative_node = acid_to_radicals_node)
        self.dag.add_transformation_node(ester_radical_and_MA_to_adduct_A, positive_node=radicalize_radical_C, negative_node = ester_radical_and_MA_to_adduct_B)
        self.dag.add_transformation_node(ester_radical_and_MA_to_adduct_B, positive_node=radicalize_radical_C)
        # acid path 
        self.dag.add_transformation_node(acid_to_radicals_node, positive_node=acid_radical_and_MA_to_adduct_A)
        self.dag.add_transformation_node(acid_radical_and_MA_to_adduct_A, positive_node=radicalize_radical_C, negative_node = acid_radical_and_MA_to_adduct_B)
        self.dag.add_transformation_node(acid_radical_and_MA_to_adduct_B, positive_node=radicalize_radical_C)
        # common path
        self.dag.add_transformation_node(radicalize_radical_C, positive_node=radicalize_radical_CCO)
        self.dag.add_transformation_node(radicalize_radical_CCO, positive_node=radicalize_adduct)
        self.dag.add_transformation_node(radicalize_adduct)
        

    def execute_TRANSFORMATION_DAG(self) -> None:
        # create initial components
        self.logger.info("Running execute_TRANSFORMATION_DAG")
        rsmi_splits = self.rsmi.split('|')
        reactant_smiles = rsmi_splits[0]
        ma_smiles = rsmi_splits[1]
        self.logger.info(f"--REACTANT_SMILES--:{reactant_smiles}")
        self.logger.info(f"--MICHAEL_ACCEPTOR_SMILES--:{ma_smiles}")

        reactant_smiles_with_H = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(reactant_smiles)))
        ma_smiles_with_H = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(ma_smiles)))

        initial_components = prepare_initial_components([reactant_smiles_with_H, ma_smiles_with_H], ['reactant', 'michael_acceptor'])
        ending_components = self.dag.run(components=initial_components)
        ignore_list: list[str] = []
        self.reaction_components = extract_and_convert_components(ending_components, to_ignore = ignore_list)
        self.logger.info("Only returning components from reaction control:")
        for component in self.reaction_components:
            self.logger.info(component.component)

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for
        each species in the reaction mechanism that has been populated

        Returns
        -------
        list[ReactionComponent]
            List of ReactionComponents objects created successfully.
        """
        return [self.solvent] + self.reaction_components

class Oxidize:
    def __init__(self, id: str, rsmi: str, logger: logging.Logger, intermediate_save_location: FileParameter[Path]) -> None:
        """
        The constructor defines the roles for the original molecule and the oxidized version.
        We take the input molecule and calculate its oxidized form by changing the charge and multiplicity.
        Then we calculate the gibbs free energy for both of these molecules.
        
        Reaction Scheme
        ---------------
        Molecule ---> oxidized molecule

        """
        self.name: str = id
        self.rsmi: str = rsmi
        self.solvent = ReactionComponent(component="solvent")
        self.reaction_components: list[ReactionComponent] = []
        self.dag: TRANSFORMATION_DAG = TRANSFORMATION_DAG()
        self.logger: logging.Logger = logger
        self.intermediate_save_location: FileParameter[Path] = intermediate_save_location
        
        self.create_components()

    def create_components(self) -> None:
        self.logger.info(f"self.rsmi:{self.rsmi}")
        neutral_mol = Chem.MolFromSmiles(self.rsmi)
        oxidized_mol = Chem.MolFromSmiles(self.rsmi)
        
        neutral_component = ReactionComponent(component='neutral', smiles=self.rsmi)
        neutral_maize_mol_obj = Loader.molecule_from_rdkit(neutral_mol)
        neutral_maize_mol_obj.name = 'neutral'
        neutral_component.structure.append(neutral_maize_mol_obj)
        neutral_charge, neutral_mult = get_chg_and_mult(neutral_mol)
        neutral_component.parameters = [neutral_charge, neutral_mult]
        self.reaction_components.append(neutral_component)

        oxidized_component = ReactionComponent(component='oxidized', smiles=self.rsmi)
        oxidized_maize_mol_obj = Loader.molecule_from_rdkit(oxidized_mol)
        oxidized_maize_mol_obj.name = 'oxidized'
        oxidized_component.structure.append(oxidized_maize_mol_obj)
        oxidized_charge, oxidized_mult = get_chg_and_mult(oxidized_mol)
        oxidized_component.parameters = [oxidized_charge+1, (neutral_mult % 2) + 1]
        self.reaction_components.append(oxidized_component)

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for
        each species in the reaction mechanism that has been populated

        Returns
        -------
        list[ReactionComponent]
            List of ReactionComponents objects created successfully.
        """
        return [self.solvent] + self.reaction_components

class OxidizeandReduce:
    def __init__(self, id: str, rsmi: str, logger: logging.Logger, intermediate_save_location: FileParameter[Path]) -> None:
        """
        The constructor defines the roles for the original molecule and the oxidized version.
        We take the input molecule and calculate its oxidized form by changing the charge and multiplicity.
        Then we calculate the gibbs free energy for both of these molecules.
        
        Reaction Scheme
        ---------------
        Molecule ---> oxidized molecule

        """
        self.name: str = id
        self.rsmi: str = rsmi
        self.solvent = ReactionComponent(component="solvent")
        self.reaction_components: list[ReactionComponent] = []
        self.dag: TRANSFORMATION_DAG = TRANSFORMATION_DAG()
        self.logger: logging.Logger = logger
        self.intermediate_save_location: FileParameter[Path] = intermediate_save_location

        self.create_components()

    def create_components(self) -> None:
        self.logger.info(f"self.rsmi:{self.rsmi}")
        neutral_mol = Chem.MolFromSmiles(self.rsmi)
        oxidized_mol = Chem.MolFromSmiles(self.rsmi)
        reduced_mol = Chem.MolFromSmiles(self.rsmi)
        
        neutral_component = ReactionComponent(component='neutral', smiles=self.rsmi)
        neutral_maize_mol_obj = Loader.molecule_from_rdkit(neutral_mol)
        neutral_maize_mol_obj.name = 'neutral'
        neutral_component.structure.append(neutral_maize_mol_obj)
        neutral_charge, neutral_mult = get_chg_and_mult(neutral_mol)
        neutral_component.parameters = [neutral_charge, neutral_mult]
        self.reaction_components.append(neutral_component)

        oxidized_component = ReactionComponent(component='oxidized', smiles=self.rsmi)
        oxidized_maize_mol_obj = Loader.molecule_from_rdkit(oxidized_mol)
        oxidized_maize_mol_obj.name = 'oxidized'
        oxidized_component.structure.append(oxidized_maize_mol_obj)
        oxidized_charge, oxidized_mult = get_chg_and_mult(oxidized_mol)
        oxidized_component.parameters = [oxidized_charge+1, (neutral_mult % 2) + 1]
        self.reaction_components.append(oxidized_component)

        reduced_component = ReactionComponent(component='reduced', smiles=self.rsmi)
        reduced_maize_mol_obj = Loader.molecule_from_rdkit(reduced_mol)
        reduced_maize_mol_obj.name = 'reduced'
        reduced_component.structure.append(reduced_maize_mol_obj)
        reduced_charge, oxidized_mult = get_chg_and_mult(reduced_mol)
        reduced_component.parameters = [reduced_charge-1, (neutral_mult % 2) + 1]
        self.reaction_components.append(reduced_component)

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for
        each species in the reaction mechanism that has been populated

        Returns
        -------
        list[ReactionComponent]
            List of ReactionComponents objects created successfully.
        """
        return [self.solvent] + self.reaction_components