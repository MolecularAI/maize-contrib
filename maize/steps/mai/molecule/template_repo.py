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
from typing import cast, Literal, Dict, Optional, Callable
from pathlib import Path
from numpy.typing import NDArray 
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R
import logging
from rdkit import Chem
from rdkit.Chem import AllChem, FragmentMatcher, Draw

from maize.core.interface import FileParameter
from maize.utilities.chem.chem import Isomer

from maize.steps.mai.molecule.compchem_utils import (
    Structure,
    Atom,
    EntryCoord,
    Loader
) 

def attach_point_constraint_violation(
    coords: NDArray[np.float32], 
    attach_index: int, 
    metal_center_coords: NDArray[np.float32]
) -> float:
    """
    Calculate how severely the attach point constraint is violated.
    
    Parameters:
    -----------
    coords : array of coordinates
    attach_index : index of the attach point in coords
    metal_center_coords : coordinates of the metal center
    
    Returns:
    --------
    float : 0 if constraint is satisfied, positive value indicating severity of violation
    """
    # Calculate distance from attach point to metal center
    attach_point = coords[attach_index]
    attach_dist = float(np.linalg.norm(attach_point - metal_center_coords))
    
    # Find minimum distance from any atom to metal center
    min_dist = float('inf')
    min_index = -1
    
    for i, point in enumerate(coords):
        dist = float(np.linalg.norm(point - metal_center_coords))
        if dist < min_dist:
            min_dist = dist
            min_index = i
    
    if min_index == attach_index:
        # Constraint satisfied
        return 0.0
    else:
        # Constraint violated - return difference in distances
        # This creates a continuous penalty that increases as the violation gets worse
        violation = float((attach_dist - min_dist) * 10.0)  # Scale factor to make this penalty significant
        return violation


def clash_score(
    coords1: NDArray[np.float32] | list[NDArray[np.float32]], 
    coords2: NDArray[np.float32] | list[NDArray[np.float32]], 
    clash_distance: float = 1.5
) -> float:
    """
    Calculate a continuous clash score between two sets of coordinates.
    Lower score is better (fewer/less severe clashes).
    
    Parameters:
    -----------
    coords1, coords2 : arrays of 3D coordinates
    clash_distance : threshold below which atoms are considered clashing
    
    Returns:
    --------
    float : clash score (0 = no clashes)
    """
    score = 0.0
    for p1 in coords1:
        for p2 in coords2:
            dist = np.linalg.norm(p1 - p2)
            if dist < clash_distance:
                score += float((clash_distance / dist - 1.0)**2)
    
    return float(score)

def perform_first_alignment(
        subst_o: NDArray[np.float32], 
        metal_cent: EntryCoord, 
        close_subst_centroid: NDArray[np.float32], 
        attach_index: int
    ) -> NDArray[np.float32]: 
    
    """
    Performs the first rotation to align the metal center, attach point, and ligand centroid.
    
    Returns the rotated coordinates.
    """
    r = np.array(metal_cent.coords, dtype=np.float32)  # vector to metal center
    s = close_subst_centroid  # vector to centroid ligand
    
    # Handle edge cases with zero vectors
    if np.linalg.norm(r) < 1e-6 or np.linalg.norm(s) < 1e-6:
        return subst_o
        
    # Compute cross product for rotation axis
    cross_v = np.cross(r, s).astype(np.float32)
    
    magnitude = np.sqrt(np.dot(cross_v, cross_v))
    if magnitude > 1e-6:
        norm_cross = cross_v / magnitude
    else:
        # Vectors are parallel or anti-parallel
        # Find any perpendicular vector
        if abs(r[0]) < abs(r[1]):
            perp = np.array([0, -r[2], r[1]], dtype=np.float32)
        else:
            perp = np.array([-r[2], 0, r[0]], dtype=np.float32)
        norm_cross = perp / np.linalg.norm(perp)
    
    # Compute rotation angle
    if np.linalg.norm(r) == 0 or np.linalg.norm(s) == 0:
        alfa = 0.0
    else:
        dot_product = np.dot(r, s) / (np.linalg.norm(r) * np.linalg.norm(s))
        # Clip to handle numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        alfa = np.arccos(dot_product)
    
    # Apply rotation
    r1 = R.from_rotvec((np.pi - alfa) * norm_cross)
    rot1 = np.array(r1.apply(subst_o), dtype=np.float32)
    
    # Verify rotation worked correctly
    c_r_cent = find_closest_centroid(rot1, rot1[attach_index])[0]
    
    # If alignment didn't work, try opposite rotation
    if not np.allclose(r, c_r_cent, atol=1e-01):
        r_neg = R.from_rotvec((alfa - np.pi) * norm_cross)
        rot1 = np.array(r_neg.apply(subst_o), dtype=np.float32)
    
    return rot1

def optimize_with_quaternions(
        rot1: NDArray[np.float32], 
        metal_cent: EntryCoord, 
        attach_point: EntryCoord, 
        new_temp_array: list[NDArray[np.float32]], 
        temp_centroid: NDArray[np.float32], 
        attach_index: int
    ) -> NDArray[np.float32]:
    """
    Optimizes the orientation using quaternions while preserving alignment.
    
    Parameters:
    -----------
    rot1 : coordinates after first rotation
    metal_cent : coordinates of metal center
    attach_point : coordinates of attachment point
    new_temp_array : template coordinates
    temp_centroid : centroid of template
    
    Returns:
    --------
    optimally rotated coordinates
    """
    # Define the axis we're allowed to rotate around (to preserve alignment)
    axis = np.array(metal_cent.coords, dtype=np.float32) - np.array(attach_point.coords, dtype=np.float32)
    if np.linalg.norm(axis) < 1e-6:
        # If points are too close, use a default axis
        axis = np.array([0, 0, 1], dtype=np.float32)
    else:
        axis = axis / np.linalg.norm(axis)
    
    def objective_function(angle: list[tuple[int, float]]) -> float:
        """
        Objective function for optimization.
        We only optimize a single angle around the fixed axis.
        """
        # Create rotation from angle around fixed axis
        rotation = R.from_rotvec(angle[0] * axis)
        rotated_coords = rotation.apply(rot1)
        
        clash_penalty = clash_score(rotated_coords, new_temp_array)
        
        # Calculate attach point constraint violation
        metal_coords = np.array(metal_cent.coords, dtype=np.float32)
        constraint_penalty = attach_point_constraint_violation(
            rotated_coords, attach_index, metal_coords)
        
        # Calculate distance metric to maximize
        distances = [np.linalg.norm(k - temp_centroid) for k in rotated_coords]
        avg_distance = sum(distances) / len(distances)
        
        # Combined objective: minimize clashes, maximize distance
        return float(clash_penalty * 10.0 + constraint_penalty * 50.0 - avg_distance)
    

    
    # Use global optimization to find best angle
    from scipy.optimize import differential_evolution
    
    # Search full 360° rotation
    bounds = [(0, 2*np.pi)]
    
    result = differential_evolution(
        objective_function,
        bounds,
        popsize=15,
        mutation=(0.5, 1.5),
        recombination=0.7,
        maxiter=50
    )
    
    # Apply best rotation
    best_rotation = R.from_rotvec(result.x[0] * axis)
    return np.array(best_rotation.apply(rot1), dtype=np.float32)

def full_quaternion_optimization(
        rot1: NDArray[np.float32], 
        new_temp_array: list[NDArray[np.float32]], 
        temp_centroid:NDArray[np.float32], 
        attach_index: int, 
        metal_cent: EntryCoord
    ) -> NDArray[np.float32]:
    """
    Fallback method that explores the full rotation space using quaternions.
    Use when constrained optimization fails to find a clash-free solution.
    """
    def objective_function(quat_params: NDArray[np.float32]) -> float:
        # Convert to unit quaternion
        quat = quat_params / np.linalg.norm(quat_params)
        
        # Apply rotation
        rotation = R.from_quat(quat)
        rotated_coords = rotation.apply(rot1)
        
        # Calculate clash penalty
        clash_penalty = clash_score(rotated_coords, new_temp_array)
        
        # Calculate attach point constraint violation
        metal_coords = np.array(metal_cent.coords, dtype=np.float32)
        constraint_penalty = attach_point_constraint_violation(
            rotated_coords, attach_index, metal_coords)
        
        # Calculate distance metric
        distances = [np.linalg.norm(k - temp_centroid) for k in rotated_coords]
        avg_distance = sum(distances) / len(distances)
        
        # Combined objective: minimize clashes, enforce constraint, maximize distance
        return float(clash_penalty * 10.0 + constraint_penalty * 50.0 - avg_distance)
    
    # Use global optimization
    from scipy.optimize import differential_evolution
    
    # Initial quaternion bounds
    bounds = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    
    result = differential_evolution(
        objective_function,
        bounds,
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.7,
        maxiter=100
    )
    
    # Normalize quaternion
    best_quat = result.x / np.linalg.norm(result.x)
    
    # Apply best rotation
    rotation = R.from_quat(best_quat)
    return np.array(rotation.apply(rot1), dtype=np.float32)

def orient_substituent(
    template_dict: list[EntryCoord],
    substituent_dict: list[EntryCoord],
    substituent_attach_center: EntryCoord,
    index: int,
) -> tuple[list[EntryCoord], list[EntryCoord], list[EntryCoord]]:
    """
    Creates ligand orientated with optimal geometry to be coordinated
    with the intermediate template.
    """
    # Prepare data (similar to your original code)
    # ...
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
    unsc_close_cent, _ = find_closest_centroid(subst_array, subst_attach_point)
    subst_o = np.array([at - subst_attach_point for at in subst_array], dtype=np.float32)
   
    # translate substituent coordinates, by putting the attach point at the origin
    or_att_pt = unsc_close_cent - subst_attach_point
    close_subst_centroid = np.array(or_att_pt, dtype=np.float32)
   
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
    for item in scaled_temp:
      
        if item.element == "Ni":
            metal_cent.element = item.element
            metal_cent.coords = item.coords
            break
    rot1 = perform_first_alignment(subst_o, metal_cent, close_subst_centroid, attach_index)
    
    # Try constrained quaternion optimization first - now with attach_index
    rotamer = optimize_with_quaternions(
        rot1, metal_cent, scaled_sub_attach_atom, new_temp_array, temp_centroid, attach_index
    )
    
    # Calculate clash score for the constrained optimization result
    constrained_clash_score = clash_score(rotamer, new_temp_array)
    
    # Check attach point constraint
    metal_coords = np.array(metal_cent.coords, dtype=np.float32)
    constraint_violation = attach_point_constraint_violation(rotamer, attach_index, metal_coords)
    
    # Determine threshold based on molecule size and current clash score
    # clash_threshold = get_clash_threshold(substituent_dict, constrained_clash_score)
    
    # Check if we need full quaternion optimization
    need_full_opt = constraint_violation > 0.1
    
    if need_full_opt:
        if constraint_violation > 0.1:
            print(f"  - Attach point constraint violated (penalty: {constraint_violation:.2f})")
        
        # Try full quaternion optimization - now with attach_index and metal_cent
        full_rotamer = full_quaternion_optimization(
            rot1, new_temp_array, temp_centroid, attach_index, metal_cent
        )
        
        # Calculate scores for full optimization
        full_clash_score = clash_score(full_rotamer, new_temp_array)
        full_constraint_violation = attach_point_constraint_violation(
            full_rotamer, attach_index, metal_coords)
        
        # Decide which result to use based on both clash score and constraint
        constrained_total_score = constrained_clash_score + constraint_violation * 10.0
        full_total_score = full_clash_score + full_constraint_violation * 10.0
        
        if full_total_score < constrained_total_score:
            rotamer = full_rotamer
            print(f"Full optimization improved total score from {constrained_total_score:.2f} to {full_total_score:.2f}")
        else:
            print(f"Full optimization did not improve total score ({full_total_score:.2f} vs {constrained_total_score:.2f})")
    
    # Final check of attach point constraint
    final_constraint_violation = attach_point_constraint_violation(
        rotamer, attach_index, metal_coords)
    
    if final_constraint_violation > 0.1:
        print(f"WARNING: Final orientation violates attach point constraint (penalty: {final_constraint_violation:.2f})")
        # You could implement additional fallback strategies here
    
    # Recreate mol dict of orientated ligand
    orientated_ligand = [EntryCoord(element=e, coords=list(c)) for e, c in zip(keys_lig, rotamer)]
    markers = [scaled_sub_attach_atom, metal_cent]
 
    return orientated_ligand, new_temp, markers

#########

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

def remove_leaving_group(
        smiles_string: str, 
        smarts_pattern: str, 
        idx_at_pt: int
    ) -> tuple[Structure, int]:
    
    """
    Remove atoms matching a SMARTS pattern from a molecule and adjust the attachment point index.
    """

    # Convert SMILES to RDKit molecule if needed
    mol = Chem.MolFromSmiles(smiles_string)
    mol.UpdatePropertyCache()
    mol = Chem.AddHs(mol)


    # Create the SMARTS pattern
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if 'OX2H' in smarts_pattern:
        pattern = Chem.AddHs(pattern)
    # Find all matches for the SMARTS pattern in the molecule
    matches = mol.GetSubstructMatches(pattern)

    # Collect all atom objects corresponding to the matched substructures
    matched_atoms = []
    for match in matches:
        for atom_idx in match:
            atom = mol.GetAtomWithIdx(atom_idx)
            matched_atoms.append([atom, atom_idx])

    
    structure = Loader.molecule_from_rdkit(mol)
    adjusted_idx = idx_at_pt
    new_lig_list = []
    
    for n, e in enumerate(structure.mol_dict):   
        include_item = True
        for match in matched_atoms:
            if e.element == match[0].GetSymbol() and n == match[1]:
                
                include_item = False
                if n < idx_at_pt:
                    adjusted_idx = adjusted_idx - 1
    
        if include_item:
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


def make_intermediate(
    template: Structure,
    index_cov: list[int] | list[str],
    ligand_coval: list[str],
    leaving_groups: list[str],
    text: str = "default",
) -> tuple[Structure, list[int], list[int] | None]:
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
    ligand_coval
        SMILES of ligand to be covalently added

    Returns
    -------
    tuple[Structure, list[int]]
        Structure of the assembled intermediate,
        Index of the atoms to fix in the sdf file
    """


    
    index_corrector = 0   
    if all(isinstance(y, int) for y in index_cov):
        index_corrector = len(index_cov)
    
    
    template.connectivity_by_distance()
    intermediate_adduct = template.mol_dict
    # get connectivity matrix. Numbers have to be scaled by removing the number of atoms that are
    # used as placeholders for interactions in the templates. These are always the first ones in 
    # the template therefore once the index is subtracted with the index corrector they will become
    # either negative or zero. Thus we can use a simple logic to select the ones we are intrested in.

    intermediate_cmatrix = [(tx - index_corrector + 1, ty - index_corrector + 1, tz) for tx, ty, tz in template.c_matrix 
                            if (tx - index_corrector + 1) > 0 and (ty - index_corrector + 1 > 0)]

    # store the index of the atom in the template to fix for future calculations
    temp_fix_indexes = len(intermediate_adduct) - index_corrector
    fix_index = list(range(temp_fix_indexes))  # list of atom index to fix. Add the atoms that are part of the template.

    m = 0
    reag2_idx_list = []
    for index, coval_bb, lg in zip(index_cov, ligand_coval, leaving_groups):
    
        cov_att_idx = smarts_id(coval_bb, lg)
        clean_ligand_cov, adjusted_index = remove_leaving_group(
            coval_bb, lg.replace("[C:1]", "").replace("[c:1]", ""), cov_att_idx
        )  # covalent ligand molecule object LeavingGroup removed

        clean_ligand_cov.connectivity_by_distance()
        clean_lig_cov_dict = clean_ligand_cov.mol_dict
        adjust_att_pt = clean_lig_cov_dict[adjusted_index]
        
        reag2_idx = None    
        if type(index) is int:
           
            if index >=  m:
                cov_adj = index-m
            else:
                cov_adj = index  
            
            orientated_ligand, new_template, cov_markers = orient_substituent(
                intermediate_adduct, clean_lig_cov_dict, adjust_att_pt, cov_adj
            )
                 
            intermediate_adduct = new_template
            met_idx = cov_markers[1]
            cov_idx = cov_markers[0]

            att_nr = None
            met_nr = None
            
            adjust_cov_idx = len(index_cov) - 1 - m
            # readjust indexes of coordination interaction point after adding ligand and removing the attach point.
            if len(index_cov) > 1:
                if m == 0:
                    adjust = len(intermediate_adduct)  
                else:
                    adjust = len(intermediate_adduct) + 1    
            else:
                adjust = len(intermediate_adduct) + 1 
                
            # append the coordinates of the ligand to the final intermediate structure
            for q in orientated_ligand:
                intermediate_adduct.append(q)
            # adjust connectivity indexes by adding the length of the existing adduct and subtracting the indices of the attach point still not removed. (Plus one for moving from 0 to 1 index.) 
            adjusted_cm = [(cmx + adjust, cmy + adjust, cmz) for cmx, cmy, cmz in clean_ligand_cov.c_matrix]
            for cm in adjusted_cm: 
                intermediate_cmatrix.append(cm)
            
            for indice, atm in enumerate(intermediate_adduct): 
                # get index for metal center
                if atm.coords == met_idx.coords and atm.element == met_idx.element:
                    if indice not in fix_index:
                        fix_index.append(indice)

                    if len(index_cov) > 1:
                        met_nr = indice - adjust_cov_idx + 1                           
                    else:
                        met_nr = indice + 1
                        
                # get index for attach point on ligand
                elif atm.coords == cov_idx.coords and atm.element == cov_idx.element:
                    reag2_idx = indice - adjust_cov_idx
                    fix_index.append(reag2_idx)
                    reag2_idx_list.append(reag2_idx)                  
                    att_nr = indice - adjust_cov_idx + 1 
                     
            if att_nr and met_nr:
                intermediate_cmatrix.append((met_nr, att_nr, 1.0))
            m += 1      
                
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
    
    
    if reag2_idx_list:
        reag2_idx_list =  [c_idx + 1 for c_idx in reag2_idx_list]

    return final_mol, fix_index_sdf, reag2_idx_list



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
    cc_carbons: list[int] | None = None
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

            
            if self.cc_carbons:
                isomer_mol.set_tag("cc_carbons", self.cc_carbons)
            

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
        def add_radical_electron(component : ReactionComponent, new_component_name : str) -> ReactionComponent:
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
    dag : Dict[
                Callable[
                    [Dict[str, list[ReactionComponent]]],
                    tuple[
                        Dict[str, list[ReactionComponent]],
                        bool
                    ]
                ],
               Dict[
                   bool,
                    Callable[
                        [Dict[str, list[ReactionComponent]]],
                        tuple[
                            Dict[str, list[ReactionComponent]],
                            bool
                        ]
                    ]]] = field(hash=False)
    start_node : Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]] | None = None

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
        current_transformation_node: Callable[[Dict[str, list[ReactionComponent]]], tuple[Dict[str, list[ReactionComponent]], bool]] | None = self.start_node
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



class NiCatCycle_CC:

    """
    Class type for the internally modified Nickel Catalytic Cycle for CC coupling for iLAB.
    Temporary class. To be merged with

    The class contains methods to obtain ReactionComponent object for each
    species and creates the appropriate reaction intermediates and transition
    states for the available templates.

    """

    def __init__(self, id: str, rsmi: str, roles: str, requested: list[str]) -> None:
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
        self.requested = requested
        self.solvent = ReactionComponent(component="solvent")
        self.aryl_X = ReactionComponent(component="aryl_halide", smarts="[c:1][F,Cl,Br,I]")
        self.alkyl_X = ReactionComponent(component="alkyl_halide", smarts="[C:1][Br,Cl,OX2H1]")
        self.arylH = ReactionComponent(component="Debrominated_aryl")
        self.alkylH = ReactionComponent(component="Debrominated_alkyl")
        self.ar_radical = ReactionComponent(component="aryl_radical", parameters=[0,2])
        self.alk_radical = ReactionComponent(component="alkyl_radical", parameters=[0,2])
        self.alk_anion = ReactionComponent(component="alkyl_anion", parameters=[-1,1])
        self.alk_cation = ReactionComponent(component="alkyl_cation", parameters=[1,1])
        self.Ni0_inter = ReactionComponent(component="Ni0_intermediate")
        self.Ni1_inter = ReactionComponent(component="Ni1_intermediate", parameters=[0, 2])
        self.Ni2_inter = ReactionComponent(component="Ni2_intermediate")
        self.Ni3_inter = ReactionComponent(component="Ni3_intermediate", parameters=[0, 2])

        self.int_geometries = ["generic"]
        self.intermediates_temp: dict[
            str, dict[str, tuple[str, list[int] | list[str]]]
        ] = {"Ni1_alkyl": {
                "generic": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_trpl_generic_alk.sdf",
                    [0])
            },
            "Ni2_OxidativeAddition": {
                "generic": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_sqpl_generic_OxAdd.sdf",
                    [0]
                   )
            },
            "Ni3_ReductiveElimination": {
                "generic": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_trbpyr_generic_RedEl.sdf",
                    [0, 1],
                ),
                "ART":(
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_trbpyr_ART_RedEl.sdf",
                    [0, 1],)
            },
        }
        self.assign_roles()


    @staticmethod
    def identify_leaving_group(mol: Chem.rdchem.Mol, component: str) -> str:
        
        """
       
        This functions finds the leaving group on alkyl and aryl halides and return
        the corresponding SMARTS string.

        Parameters
        ----------
        mol
            Rdkit mol object for the structure to check.
        component
            defines whether to look for aryl or alkyl leaving groups.
    
        Returns
        -------
        str
            Smarts pattern of the leaving group                
        """

        smarts_patterns =  {"aryl_halides": ["[c:1][Br]", "[c:1][Cl]", "[c:1][I]"],
                            "alkyl_halides": ["[C:1][Br]", "[C:1][OX2H1]", "[C:1][Cl]", "[C:1][#5]1(-[#8]-[#6](-[#6](-[#8]-1)(-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H])(-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H])"]} 
        
        leaving_group = ""
        for smarts_string in smarts_patterns[component]:
            smarts = Chem.MolFromSmarts(smarts_string)
            mh =Chem.AddHs(mol)
            if mh.HasSubstructMatch(smarts):
                leaving_group = smarts_string
                break
        
        return leaving_group



    def assign_roles(self) -> None:
        """
        This function contains the logic to assign the various roles specific to the reaction.
        It populates also the intermediates by calling the the create_intermediate function.
        """

        separ = self.rsmi.split(">>")
        left_hand = separ[0]

        # reactant 1
        reac1_smiles = left_hand.split(".")[0]
        reac1_rdmol = Chem.MolFromSmiles(reac1_smiles)
        mol_reac1 = Loader.molecule_from_rdkit(reac1_rdmol)
        mol_reac1.name = self.aryl_X.component
        self.aryl_X.smiles = reac1_smiles
        self.aryl_X.smarts = self.identify_leaving_group(reac1_rdmol, 'aryl_halides')
        self.aryl_X.mw = mol_reac1.get_mw()
        self.aryl_X.structure.append(mol_reac1)
        self.aryl_X.parameters = get_chg_and_mult(reac1_rdmol)

        # reactant 2
        reac2_smiles = left_hand.split(".")[1]
        reac2_rdmol = Chem.MolFromSmiles(reac2_smiles)
        mol_reac2 = Loader.molecule_from_rdkit(reac2_rdmol)
        mol_reac2.name = self.alkyl_X.component
        self.alkyl_X.smiles = reac2_smiles
        self.alkyl_X.smarts = self.identify_leaving_group(reac2_rdmol, 'alkyl_halides')
        self.alkyl_X.mw = mol_reac2.get_mw()
        self.alkyl_X.structure.append(mol_reac2)
        self.alkyl_X.parameters = get_chg_and_mult(reac2_rdmol)

        # Aryl component decompositions
        ar_smarts = "[c:1]"
        ar_rdmol = Chem.MolFromSmiles(reac1_smiles)
        aryl_decomp_SMARTS = f"{self.aryl_X.smarts}>>{ar_smarts}"
        smirks_object = AllChem.ReactionFromSmarts(aryl_decomp_SMARTS)
        smirks_object.RunReactantInPlace(ar_rdmol)
    
        ar_rad_mol = Loader.molecule_from_rdkit(ar_rdmol) # aryl radical
        ar_rad_mol.name = self.ar_radical.component
        self.ar_radical.structure.append(ar_rad_mol)

        # protodehalogenated aryl halide
        debr_ar = Chem.AddHs(ar_rdmol)
        Chem.SanitizeMol(debr_ar)
        Chem.RemoveStereochemistry(debr_ar)
        debr_ar_rdmol = Chem.RemoveHs(debr_ar)
        debr_ar_mol = Loader.molecule_from_rdkit(debr_ar_rdmol) 
        debr_ar_mol.name = self.arylH.component
        self.arylH.structure.append(debr_ar_mol)
        self.arylH.parameters = get_chg_and_mult(debr_ar_rdmol)



        # Alkyl component decomposition
        
        alk_smarts = "[C:1]"
        alk_rdmol = Chem.AddHs(Chem.MolFromSmiles(reac2_smiles))
        print(self.alkyl_X.smarts)

        reaction_SMARTS = f"{self.alkyl_X.smarts}>>{alk_smarts}"
        smirks_object = AllChem.ReactionFromSmarts(reaction_SMARTS)
        smirks_object.RunReactantInPlace(alk_rdmol)
    
        alkrad_mol = Loader.molecule_from_rdkit(alk_rdmol) # alkyl radical
        alkrad_mol.name = self.alk_radical.component
        self.alk_radical.structure.append(alkrad_mol)

        alkan_mol = Loader.molecule_from_rdkit(alk_rdmol)  # alkyl anion
        alkan_mol.name = self.alk_anion.component
        self.alk_anion.structure.append(alkan_mol)
 
        alkcat_mol = Loader.molecule_from_rdkit(alk_rdmol)  # alkyl cation
        alkcat_mol.name = self.alk_cation.component
        self.alk_cation.structure.append(alkcat_mol)
        
        # Protodehalogenated alkyl halide
        alkH_rdmol = copy.deepcopy(alk_rdmol) 
        alkH_rdmol.UpdatePropertyCache()
        for atom in alkH_rdmol.GetAtoms():
            atom.UpdatePropertyCache()
            if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3 and atom.GetSymbol() == 'C':    
                if atom.GetTotalValence() != 4:
                    atom.SetNumExplicitHs(1 + atom.GetTotalNumHs()) 
        alkH_mol = Loader.molecule_from_rdkit(alkH_rdmol)
        alkH_mol.name = self.alkylH.component
        self.alkylH.structure.append(alkH_mol)


        # solvent
        if self.others:
            solv_smiles = self.others.split(".")[-1]
            solv_rdmol = Chem.MolFromSmiles(solv_smiles)
            mol_solv = Loader.molecule_from_rdkit(solv_rdmol)
            mol_solv.name = self.solvent.component
            self.solvent.smiles = solv_smiles
            self.solvent.mw = mol_solv.get_mw()
            self.solvent.structure.append(mol_solv)
            self.solvent.parameters = get_chg_and_mult(solv_rdmol)

       
        # Ni1 inter
        self.Ni1_inter.structure, self.Ni1_inter.constraints, self.Ni1_inter.cc_carbons = self.create_intermediates("Ni1_alkyl")      

        # Ni3 inter
        self.Ni3_inter.structure, self.Ni3_inter.constraints,  self.Ni3_inter.cc_carbons = self.create_intermediates("Ni3_ReductiveElimination")
       


    @staticmethod
    def retrieve_template(
        int: dict[str, tuple[str, list[str] | list[int]]], geom: str
    ) -> tuple[Structure, list[int] | list[str]]:
        """
        This function takes as input an intermediate and a type of geometry
        and returns the respective template as Structure object and the
        associated indexes for covalent and coordinate interactions.

        """

        template_repo = int
        filename = template_repo[geom][0]
        index_cov = template_repo[geom][1]

        template_load = Loader(filename)
        temp_molecule = template_load.molecule()

        return temp_molecule, index_cov

    def create_intermediates(
        self, intermediate_template: str
    ) -> tuple[list[Structure], list[list[int]], list[int] | None]:
        """
        This function creates the proper intermediate molecule
        object and the associated atom to be constrained.

        Parameters
        ----------
        intermediate_template
            template structure for the intermediate molecule
        
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
        

        for geom in self.int_geometries:
            temp_mol, cov_idxs = NiCatCycle_CC.retrieve_template(
                self.intermediates_temp[intermediate_template], geom
            )
          
            test_iter, atms_to_fix, cc_carbons = make_intermediate(
                temp_mol,
                cov_idxs,
                [self.alkyl_X.smiles, self.aryl_X.smiles],
                [self.alkyl_X.smarts, self.aryl_X.smarts],
                "{}_{}".format(intermediate_template, geom),
            )

            
            intermediates_list.append(test_iter)
            constraints_list.append(atms_to_fix)
            
        return intermediates_list, constraints_list, cc_carbons

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for each
        species in the reaction mechanism that has been populated

        """

        available: list[ReactionComponent] = []
        std_reaction_components = [
            self.aryl_X,
            self.alkyl_X,
            self.arylH,
            self.alkylH,
            self.ar_radical,
            self.alk_radical,
            self.Ni0_inter,
            self.Ni1_inter,
            self.Ni2_inter,
            self.Ni3_inter
        ]
        if 'all' in self.requested:
            available = [component for component in std_reaction_components]
        else:
            for f in std_reaction_components:
                if f.component in self.requested:
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

    def __init__(self, id: str, rsmi: str, logger: logging.Logger) -> None:
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
        self.reaction_components: list[ReactionComponent] = []
        self.dag: TRANSFORMATION_DAG = TRANSFORMATION_DAG()
        self.logger: logging.Logger = logger

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
    def __init__(self, id: str, rsmi: str, logger: logging.Logger) -> None:
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
    def __init__(self, id: str, rsmi: str, logger: logging.Logger) -> None:
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
     
class NiCatCycle_CC2:

    """
    Class type for the internally modified Nickel Catalytic Cycle for CC coupling for iLAB.
    Temporary class. To be merged with

    The class contains methods to obtain ReactionComponent object for each
    species and creates the appropriate reaction intermediates and transition
    states for the available templates.

    """

    def __init__(self, id: str, rsmi: str, roles: str, requested: list[str]) -> None:
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
        self.requested = requested
        self.solvent = ReactionComponent(component="solvent")
        self.aryl_X = ReactionComponent(component="aryl_halide", smarts="[c:1][F,Cl,Br,I]")
        self.alkyl_X = ReactionComponent(component="alkyl_halide", smarts="[C:1][Br,Cl,OX2H1]")
        self.arylH = ReactionComponent(component="Debrominated_aryl")
        self.alkylH = ReactionComponent(component="Debrominated_alkyl")
        self.ar_radical = ReactionComponent(component="aryl_radical", parameters=[0,2])
        self.alk_radical = ReactionComponent(component="alkyl_radical", parameters=[0,2])
        self.alk_anion = ReactionComponent(component="alkyl_anion", parameters=[-1,1])
        self.alk_cation = ReactionComponent(component="alkyl_cation", parameters=[1,1])
        self.Ni0_inter = ReactionComponent(component="Ni0_intermediate")
        self.Ni1_inter = ReactionComponent(component="Ni1_intermediate", parameters=[0, 2])
        self.Ni2_inter = ReactionComponent(component="Ni2_intermediate")
        self.Ni3_inter = ReactionComponent(component="Ni3_intermediate", parameters=[0, 2])

        self.int_geometries = ["ART"]
        self.intermediates_temp: dict[
            str, dict[str, tuple[str, list[int] | list[str]]]
        ] = {"Ni1_alkyl": {
                "ART": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiI_trpl_ART_alk.sdf",
                    [0],
                )},
            "Ni2_OxidativeAddition": {
                "generic": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiII_sqpl_generic_OxAdd.sdf",
                    [0],
                   )
            },
            "Ni3_ReductiveElimination": {
                "generic": (
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_trbpyr_generic_RedEl.sdf",
                    [0, 1],
                ),
                "ART":(
                    f"{Path(__file__).parent}/templates/CC_templ/NiIII_trbpyr_ART_RedEl.sdf",
                    [0, 1],)
            },
        }
        self.assign_roles()


    @staticmethod
    def identify_leaving_group(mol: Chem.rdchem.Mol, component: str) -> str:
        
        """
       
        This functions finds the leaving group on alkyl and aryl halides and return
        the corresponding SMARTS string.

        Parameters
        ----------
        mol
            Rdkit mol object for the structure to check.
        component
            defines whether to look for aryl or alkyl leaving groups.
    
        Returns
        -------
        str
            Smarts pattern of the leaving group                
        """

        smarts_patterns =  {"aryl_halides": ["[c:1][Br]", "[c:1][Cl]", "[c:1][I]"],
                            "alkyl_halides": ["[C:1][Br]", "[C:1][OX2H1]", "[C:1][Cl]", "[C:1][#5]1(-[#8]-[#6](-[#6](-[#8]-1)(-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H])(-[#6](-[H])(-[H])-[H])-[#6](-[H])(-[H])-[H])"]} 

        leaving_group = ""
        for smarts_string in smarts_patterns[component]:
            smarts = Chem.MolFromSmarts(smarts_string)
            mh = Chem.AddHs(mol)
            if mh.HasSubstructMatch(smarts):
                leaving_group = smarts_string
                break
        
        return leaving_group



    def assign_roles(self) -> None:
        """
        This function contains the logic to assign the various roles specific to the reaction.
        It populates also the intermediates by calling the the create_intermediate function.
        """

        separ = self.rsmi.split(">>")
        left_hand = separ[0]

        # reactant 1
        reac1_smiles = left_hand.split(".")[0]
        reac1_rdmol = Chem.MolFromSmiles(reac1_smiles)
        mol_reac1 = Loader.molecule_from_rdkit(reac1_rdmol)
        mol_reac1.name = self.aryl_X.component
        self.aryl_X.smiles = reac1_smiles
        self.aryl_X.smarts = self.identify_leaving_group(reac1_rdmol, 'aryl_halides')
        self.aryl_X.mw = mol_reac1.get_mw()
        self.aryl_X.structure.append(mol_reac1)
        self.aryl_X.parameters = get_chg_and_mult(reac1_rdmol)

        # reactant 2
        reac2_smiles = left_hand.split(".")[1]
    

        reac2_rdmol = Chem.MolFromSmiles(reac2_smiles)
        mol_reac2 = Loader.molecule_from_rdkit(reac2_rdmol)
        mol_reac2.name = self.alkyl_X.component
        self.alkyl_X.smiles = reac2_smiles
        self.alkyl_X.smarts = self.identify_leaving_group(reac2_rdmol, 'alkyl_halides')
        self.alkyl_X.mw = mol_reac2.get_mw()
        self.alkyl_X.structure.append(mol_reac2)
        self.alkyl_X.parameters = get_chg_and_mult(reac2_rdmol)

        # Aryl component decompositions
        ar_smarts = "[c:1]"
        ar_rdmol = Chem.MolFromSmiles(reac1_smiles)
        aryl_decomp_SMARTS = f"{self.aryl_X.smarts}>>{ar_smarts}"
        smirks_object = AllChem.ReactionFromSmarts(aryl_decomp_SMARTS)
        smirks_object.RunReactantInPlace(ar_rdmol)
    
        ar_rad_mol = Loader.molecule_from_rdkit(ar_rdmol) # aryl radical
        ar_rad_mol.name = self.ar_radical.component
        self.ar_radical.structure.append(ar_rad_mol)

        # protodehalogenated aryl halide
        debr_ar = Chem.AddHs(ar_rdmol)
        Chem.SanitizeMol(debr_ar)
        Chem.RemoveStereochemistry(debr_ar)
        debr_ar_rdmol = Chem.RemoveHs(debr_ar)
        debr_ar_mol = Loader.molecule_from_rdkit(debr_ar_rdmol) 
        debr_ar_mol.name = self.arylH.component
        self.arylH.structure.append(debr_ar_mol)
        self.arylH.parameters = get_chg_and_mult(debr_ar_rdmol)



        # Alkyl component decomposition
        
        alk_smarts = "[C:1]"
        alk_rdmol = Chem.AddHs(Chem.MolFromSmiles(reac2_smiles))
        reaction_SMARTS = f"{self.alkyl_X.smarts}>>{alk_smarts}"
        smirks_object = AllChem.ReactionFromSmarts(reaction_SMARTS)
        smirks_object.RunReactantInPlace(alk_rdmol)
    
        alkrad_mol = Loader.molecule_from_rdkit(alk_rdmol) # alkyl radical
        alkrad_mol.name = self.alk_radical.component
        self.alk_radical.structure.append(alkrad_mol)

        alkan_mol = Loader.molecule_from_rdkit(alk_rdmol)  # alkyl anion
        alkan_mol.name = self.alk_anion.component
        self.alk_anion.structure.append(alkan_mol)
 
        alkcat_mol = Loader.molecule_from_rdkit(alk_rdmol)  # alkyl cation
        alkcat_mol.name = self.alk_cation.component
        self.alk_cation.structure.append(alkcat_mol)
        
        # Protodehalogenated alkyl halide
        alkH_rdmol = copy.deepcopy(alk_rdmol) 
        alkH_rdmol.UpdatePropertyCache()
        for atom in alkH_rdmol.GetAtoms():
            atom.UpdatePropertyCache()
            if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3 and atom.GetSymbol() == 'C':    
                if atom.GetTotalValence() != 4:
                    atom.SetNumExplicitHs(1 + atom.GetTotalNumHs()) 
        alkH_mol = Loader.molecule_from_rdkit(alkH_rdmol)
        alkH_mol.name = self.alkylH.component
        self.alkylH.structure.append(alkH_mol)


        # solvent
        if self.others:
            solv_smiles = self.others.split(".")[-1]
            solv_rdmol = Chem.MolFromSmiles(solv_smiles)
            mol_solv = Loader.molecule_from_rdkit(solv_rdmol)
            mol_solv.name = self.solvent.component
            self.solvent.smiles = solv_smiles
            self.solvent.mw = mol_solv.get_mw()
            self.solvent.structure.append(mol_solv)
            self.solvent.parameters = get_chg_and_mult(solv_rdmol)

        # Ni1 inter
        self.Ni1_inter.structure, self.Ni1_inter.constraints, self.Ni1_inter.cc_carbons = self.create_intermediates("Ni1_alkyl")      
        # Ni3 inter
        self.Ni3_inter.structure, self.Ni3_inter.constraints, self.Ni3_inter.cc_carbons = self.create_intermediates("Ni3_ReductiveElimination")
       


    @staticmethod
    def retrieve_template(
        int: dict[str, tuple[str, list[int] | list[str]]], geom: str
    ) -> tuple[Structure, list[int] | list[str]]:
        """
        This function takes as input an intermediate and a type of geometry
        and returns the respective template as Structure object and the
        associated indexes for covalent and coordinate interactions.

        """

        template_repo = int
        filename = template_repo[geom][0]
        index_cov = template_repo[geom][1]

        template_load = Loader(filename)
        temp_molecule = template_load.molecule()

        return temp_molecule, index_cov

    def create_intermediates(
        self, intermediate_template: str
    ) -> tuple[list[Structure], list[list[int]], list[int] | None]:
        """
        This function creates the proper intermediate molecule
        object and the associated atom to be constrained.

        Parameters
        ----------
        intermediate_template
            template structure for the intermediate molecule
        
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
   

        for geom in self.int_geometries:
            temp_mol, cov_idxs = NiCatCycle_CC.retrieve_template(
                self.intermediates_temp[intermediate_template], geom
            )
          
            test_iter, atms_to_fix, cc_carbons = make_intermediate(
                temp_mol,
                cov_idxs,
                [self.alkyl_X.smiles, self.aryl_X.smiles],
                [self.alkyl_X.smarts, self.aryl_X.smarts],
                "{}_{}".format(intermediate_template, geom),
            )

            
            intermediates_list.append(test_iter)
            constraints_list.append(atms_to_fix)
            
        return intermediates_list, constraints_list, cc_carbons

    def return_reaction_components(self) -> list[ReactionComponent]:
        """
        Simple functions that returns the ReactionComponent objects for each
        species in the reaction mechanism that has been populated

        """

        available: list[ReactionComponent] = []
        std_reaction_components = [
            self.aryl_X,
            self.alkyl_X,
            self.arylH,
            self.alkylH,
            self.ar_radical,
            self.alk_radical,
            self.Ni0_inter,
            self.Ni1_inter,
            self.Ni2_inter,
            self.Ni3_inter
        ]
        if 'all' in self.requested:
            available = [component for component in std_reaction_components]
        else:
            for f in std_reaction_components:
                if f.component in self.requested:
                    if f.structure:
                        available.append(f)

        return available