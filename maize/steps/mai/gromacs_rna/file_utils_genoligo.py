"""Helper Functions to perform structure manipulations and a Node to save the output pdbs"""

from pathlib import Path
import shutil
from typing import Any, Generic, TypeVar
from maize.core.node import Node
from maize.core.interface import (
    Input,
    FileParameter,
    Flag,
)


def replace_res(bb: str, replace: Any, str_obmol: Any, mod_obmol: Any, chain: Any) -> Any:
    """Setting residue names and numbers and atom names for them modified residues"""
    from openbabel import openbabel as ob

    final_mol = ob.OBMol()
    current_res = ob.OBResidue()

    # Codes from the pnab input file translated to residue names in gromacs topology. Names are in correspondence to siRNA database
    dict_resnames = {
        "pr": " ",
        "sr": "P",
        "ss": "P",
        "pm": "M",
        "sm": "B",
        "pf": "F",
        "sf": "H",
        "pg": "G",
        "pd": "D",
        "sd": "S",
        "rd": "S",
        "pe": "E",
        "se": "O",
        "pl": "L",
        "sl": "N",
        "rl": "N",
    }

    atom_names = []
    residue_names = []
    residue_numbers = []

    for i, (res_str, res_mod) in enumerate(
        zip(ob.OBResidueIter(str_obmol), ob.OBResidueIter(mod_obmol))
    ):
        residue_number = res_str.GetNum()

        if residue_number in replace:
            current_res = res_mod
            current_res.SetName(dict_resnames[bb] + current_res.GetName().strip())
        else:
            current_res = res_str
            current_res.SetName(current_res.GetName().strip())

        for atom in ob.OBResidueAtomIter(current_res):
            final_mol.AddAtom(atom)
            current_atom = final_mol.GetAtom(final_mol.NumAtoms())
            current_atom.SetResidue(current_res)
            atom_names.append(current_res.GetAtomID(atom))
            residue_names.append(current_res.GetName())
            residue_numbers.append(current_res.GetNum())

    for i, atom in enumerate(ob.OBMolAtomIter(final_mol)):
        atom.GetResidue().SetAtomID(atom, atom_names[i])
        atom.GetResidue().SetName(residue_names[i])
        atom.GetResidue().SetNum(residue_numbers[i])
        atom.GetResidue().SetHetAtom(atom, False)
        atom.GetResidue().SetChain(chain)

    return final_mol


def keep_p5(path: Path, mol: Any) -> Any:
    """Keep the phosphate group only at the 5' end of the strand and rename the atoms and residue"""
    from openbabel import openbabel as ob
    import parmed as pmd

    conv = ob.OBConversion()
    conv.WriteFile(mol, f"{path}/temp.pdb")
    temp = pmd.load_file(f"{path}/temp.pdb")

    for atom in temp.residues[0]:
        if atom.name == "O5T":
            atom.name = "O3P"

    res_5term = "5" + temp.residues[0].name.rstrip(temp.residues[0].name[-1])
    temp.residues[0].name = res_5term

    temp.strip(f":1@H5T")
    temp.write_pdb(f"{path}/temp.pdb", use_hetatoms=False, increase_tercount=False)
    conv.ReadFile(mol, f"{path}/temp.pdb")

    return mol


def remove_p5(path: Path, mol: Any) -> Any:
    """Remove the phosphate group at the 5' end and rename the atoms and residues"""
    from openbabel import openbabel as ob
    import parmed as pmd

    conv = ob.OBConversion()
    conv.WriteFile(mol, f"{path}/temp.pdb")
    temp = pmd.load_file(f"{path}/temp.pdb")
    for atom in temp.residues[0]:
        if atom.name == "P":
            atom.name = "HO5'"
            atom.atom_type = "H"
            atom.atomic_number = 1

    temp.strip(":1@OP1,OP2,O1P,O2P,O5T,H5T,OP,SP")
    temp.write_pdb(f"{path}/temp.pdb", use_hetatoms=False, increase_tercount=False)
    conv.ReadFile(mol, f"{path}/temp.pdb")

    return mol


def minimize(mol: Any, num_residues_chain: int) -> Any:
    """Energy minimization"""
    from openbabel import openbabel as ob

    # Add bonds between residues
    # Get bonding atom for first residue
    res = mol.GetResidue(0)
    for atom in ob.OBResidueAtomIter(res):
        atom_name = res.GetAtomID(atom)
        if atom_name.strip() == "O3'":
            bond_index1 = atom.GetIdx()
            break

    # Loop over the remaining residues and form bonds between the residues
    for i in range(1, num_residues_chain):
        res = mol.GetResidue(i)
        for atom in ob.OBResidueAtomIter(res):
            atom_name = res.GetAtomID(atom)
            if atom_name.strip() == "O3'":
                bond_index1_new = atom.GetIdx()
            elif atom_name.strip() == "P":
                bond_index2 = atom.GetIdx()

        mol.AddBond(bond_index1, bond_index2, 1)
        bond_index1 = bond_index1_new

    # Optimize the backbone and fix all the base atoms
    constraints = ob.OBFFConstraints()

    for residue in ob.OBResidueIter(mol):
        for atom in ob.OBResidueAtomIter(residue):
            atom_name = residue.GetAtomID(atom)
            if "P" not in atom_name and "'" not in atom_name:
                constraints.AddAtomConstraint(atom.GetIdx())

    forcefield = ob.OBForceField.FindForceField("GAFF")
    forcefield.Setup(mol, constraints)
    forcefield.SteepestDescent(200)
    forcefield.GetCoordinates(mol)

    return mol


T = TypeVar("T")


class SavePdb(Node, Generic[T]):
    """
    SavePdb class is a Node that receives a dictionary of files and saves them to one common destination folder. The dictionary keys are of type str and the values are of type Path.
    """

    inp: Input[dict[str, Path]] = Input(mode="copy")
    """Dictionary of files input (keys are str, values are either Path objects)."""

    destination: FileParameter[Path] = FileParameter(exist_required=False)
    """The destination folder where files will be saved."""

    overwrite: Flag = Flag(default=False)
    """If True, will overwrite any previously existing file in the destination."""

    def run(self) -> None:
        files = self.inp.receive()
        self.logger.info("Files received %s", files)

        files = {k: file.absolute() for k, file in files.items()}
        dest = self.destination.filepath
        self.logger.info("Parent destination is %s", dest)

        if not dest.is_dir():
            raise ValueError(f"Destination '{dest}' must be a directory")

        files = {k: file.absolute() for k, file in files.items()}

        for k, file in files.items():
            dest_path = dest.absolute() / file.name

            if not dest_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copyfile(file, dest_path)
