from pathlib import Path
import string
import shutil
import os
import re
from typing import Any, Optional, Generic, TypeVar
from rdkit import Chem
from maize.core.node import Node, LoopedNode
from maize.core.interface import (
    Input,
    FileParameter,
    Flag,
)
import pytest


class MDPFileParser:
    """
    MDPFileParser class for parsing and updating numeric values in an MDP configuration file.

    """

    def __init__(self, filepath: Path, replacements: Optional[dict[str, str | int]] = None) -> None:
        """
        Parameters
        ----------
        filepath
            The path to the MDP configuration file.
        replacements
            Keyword-replacement mapping for numeric values in the MDP file.

        """
        self.filepath = filepath
        self.logger = None
        self.replacements = replacements or {}

    def extract_number(self, lines: list[str], keyword: str) -> int:
        """
        The caller of this function can check the return value to determine whether
        the keyword was found and what the associated number is. If the return value
        is 0, it means the keyword was not found in the input lines

        Prameters
        ---------
        lines
            Extracts a numeric value associated with a keyword in the MDP file lines.
        keyword
            Keywork to match
        Returns
        -------
        int
            Returns a number
        """
        pattern = rf'({keyword.replace("-", "[- ]")}\s*=\s*)(\d+)\s*'
        for index, line in enumerate(lines):
            match = re.search(pattern, line)
            if match:
                original_format = match.group(1)
                number = int(match.group(2))
                replacement = self.replacements.get(keyword)
                if replacement is not None:
                    lines[index] = re.sub(pattern, f"{original_format}{replacement}", line)
                return number
        return 0

    def parse(self) -> None:
        """
        Parses the MDP configuration file, extracts numeric values, and updates
        class attributes. This method reads the file, extracts specific numeric
        values, and updates class attributes accordingly. It also saves the
        modified lines back to the file.

        """
        with open(self.filepath, "r") as f:
            lines = [line.rstrip() for line in f.readlines()]

        self.num_steps = self.extract_number(lines, "nsteps")
        self.chk_steps = self.extract_number(lines, "nstcheckpoint")
        self.last_nstxout_value = self.extract_number(lines, "nstxout")
        self.last_nstvout_value = self.extract_number(lines, "nstvout")
        self.last_nstfout_value = self.extract_number(lines, "nstfout")

        last_nstxtcout = None  # Keep track of the last value for nstxtcout or nstxout-compressed
        for keyword in ["nstxtcout", "nstxout-compressed"]:
            value = self.extract_number(lines, keyword)
            if value is not None:
                last_nstxtcout = value

        self.last_nstxtcout_value = last_nstxtcout

        # Save the modified lines back to the file
        with open(self.filepath, "w") as f:
            f.write("\n".join(lines))

        # Re-run the extraction to update variable values
        self.extract_values(lines)

    def extract_values(self, lines: list[str]) -> None:
        """
        Extract numeric values from a list of text lines and update class attributes.
        This method re-runs the extraction process to update class attributes based
        on the provided lines.

        Parameters
        --------
        lines
            Extracts numeric values from a list of text lines and updates class attributes.
            This method re-runs the extraction process to update class attributes based on
            the provided lines.

        """
        self.num_steps = self.extract_number(lines, "nsteps")
        self.chk_steps = self.extract_number(lines, "nstcheckpoint")
        self.last_nstxout_value = self.extract_number(lines, "nstxout")
        self.last_nstvout_value = self.extract_number(lines, "nstvout")
        self.last_nstfout_value = self.extract_number(lines, "nstfout")

        last_nstxtcout = None

        for keyword in ["nstxtcout", "nstxout-compressed"]:
            value = self.extract_number(lines, keyword)
            if value is not None:
                last_nstxtcout = value

        self.last_nstxtcout_value = last_nstxtcout


def read_sdf_and_save_mols(input_sdf: Path, output_directory: Path) -> list[Path]:
    """
    This function reads molecules from the input SDF file, processes each molecule,
    and saves them to separate SDF files in the specified output directory. The filenames
    are derived from the content of the first line in the SDF block.

    Parameters
    ----------
    input_sdf
        Path to the input SDF file.
    output_directory
        Directory where the output SDF files will be saved.

    Returns
    -------
    List[Path]
        List of Path objects representing the paths of the saved SDF files.

    """
    saved_paths = []
    suppl = Chem.SDMolSupplier(input_sdf)

    for mol in suppl:
        if mol is not None:
            filename = mol.GetProp("_Name")
            output_file = Path(output_directory) / f"{filename}.sdf"
            writer = Chem.SDWriter(str(output_file))
            writer.write(mol)
            writer.close()

            saved_paths.append(output_file)

    return saved_paths


def add_prefix_to_filename(
    original_file_path: Path, ligand_name: str, replica_num: int, mdtp: str
) -> Path:
    """
    Add a custom prefix to a filename and return the new Path object.

    Parameters
    ----------
    original_file_path
        The original file's path including filename.
    ligand_name
        The ligand name to be used as a prefix.
    replica_num
        The replica number to be included in the filename.
    mdtp
        A description or identifier to be included in the filename.

    Returns
    -------
    Path
        A new Path object representing the modified filename with the added prefix.

    """
    # Extract the file extension (if any) from the original filename
    file_dir = original_file_path.parent
    file_name = original_file_path.name
    # Construct the new filename with the desired prefix
    new_filename = Path(file_dir / f"{ligand_name}_replica{replica_num}_{mdtp}_{file_name}")

    return new_filename


def convert_data_dict(data_dict: dict[tuple[int, str], Path]) -> dict[tuple[int, str], Path]:
    """
    Convert filenames in the given data dictionary to a simplified format,
    keeping the directory structure unchanged.

    Parameters
    ----------
    data_dict
        Input dictionary with keysas tuples representing (replica_num, ligand_name)
        and values as Path objects representing file paths

    Returns
    -------
    dict[tuple[int, str], Path]
        New dictionary with the same keys and updated file paths where filenames
        have been simplified (topol.tpr as an example).

    """
    new_data_dict = {}
    for key, file_path in data_dict.items():
        file_name = file_path.name

        # Extract the unique part of the filename
        unique_part = "_".join(file_name.split("_")[3:])

        # Create the new Path with the updated filename
        new_path = file_path.parent / Path(unique_part)
        shutil.move(file_path, new_path)

        # Update the new data_dict
        new_data_dict[key] = new_path

    return new_data_dict


def extract_filename(
    file_path: Path, ligand_name: str, replica_num: int, valid_mdptypes: list[str]
) -> Path | None:
    """
    Extract the remaining part of a filename that matches a specific pattern.
    This function is used to find and extract the remaining part of a filename
    that matches a specific pattern based on the provided ligand name, replica number,
    and a list of valid mdptype values.

    Parameters
    -----------
    data_dict
        A dictionary containing key-value pairs mapping (replica_num, ligand_name) to file paths
    ligand_name
        The ligand name to search for in the filename
    replica_num
        The replica number to search for in the filename
    valid_mdptypes
        A list of valid mdptype values to match against the filename

    Returns
    --------
    Path|None
        The remaining part of the filename that matches the specified pattern, or None if not found.

    """
    file_name = file_path.name

    # Define a list of valid mdptype values
    # Check if the filename matches any of the expected patterns
    rest_of_filename = None
    for mdptype_option in valid_mdptypes:
        expected_pattern = f"{ligand_name}_replica{replica_num}_{mdptype_option}_"
        if file_name.startswith(expected_pattern):
            rest_of_filename = Path(file_name[len(expected_pattern) :])
            break
        else:
            rest_of_filename = None

    return rest_of_filename


def process_files(
    lig_rep_tp_dirs: list[Path], mdtp: str, filename: str
) -> dict[tuple[int, str], Path]:
    """
    Rename and move files with a specific filename pattern to a new format.
    This function processes a list of directories and renames/moves files with a
    specified filename pattern to a new format. The new filename format includes
    a prefix based on ligand name, replica number, and mdptype.

    Parameters
    ----------
    lig_rep_tp_dirs
        A list of directories to search for files.
    mdtp
        The mdptype to be included in the new filename.
    filename
        The filename pattern to match in the directories.

    Returns
    -------
    dict[tuple[int, str], Path]
        A dictionary that maps (replica_num, ligand_name) to the new file paths.

    """
    files = [file for lig_rep_tp in lig_rep_tp_dirs for file in lig_rep_tp.glob(filename)]
    dict_files = {}

    for file in files:
        ligand_name = file.parts[-4]
        replica_num = int(file.parts[-3].split("Replica")[-1])
        name_with_prefix = add_prefix_to_filename(file, ligand_name, replica_num, mdtp)
        shutil.move(file, name_with_prefix)
        dict_files[(replica_num, ligand_name)] = name_with_prefix.absolute()

    return dict_files


def generate_replicas(
    num_replicas: int, file: Path, ligand_name: str, mdtp: str
) -> dict[tuple[int, str], Path]:
    """
    Generate and rename multiple replica files based on a template file.
    This function generates a specified number of replica files by making copies
    of a template file and renaming them with a custom prefix based on the ligand
    name, replica number, and mdptype. The original template file is removed.

    Parameters
    ----------
    num_replicas
        The number of replica files to generate.
    file
        The template file to be used as a basis for generating replicas.
    ligand_name
        The ligand name to include in the filenames.
    mdtp
        The mdptype to include in the filenames.

    Returns
    -------
    dict[tuple[int, str], Path]
        A dictionary that maps (replica_num, ligand_name) to the new file paths.

    """
    dict_file = {}
    for i in range(num_replicas):
        f_parent = file.parent
        f_name = file.name
        name_with_prefix = Path(f_parent) / f"{ligand_name}_replica{i+1}_{mdtp}_{f_name}"
        shutil.copy(file, name_with_prefix)
        dict_file.update({(i + 1, ligand_name): name_with_prefix})

    os.remove(file)

    return dict_file


def get_index(index_file: Path, var: str) -> int | None:
    """
    Retrieve the index of a specified variable within an index file.
    This function reads the content of the given index file, which is assumed to
    have a format with variable names enclosed in square brackets, like '[variable_name]'.
    It creates a dictionary mapping variable names to their corresponding indices.

    Parameters
    ----------
    index_file
        The path to the index file containing variable names.
    var
        The variable name for which the index is sought.

    Returns
    -------
    int | None
        The index of the specified variable. Returns `None` if the variable is not found.

    """
    with open(index_file, "r") as f:
        lines = f.readlines()
        index_dict = {}
        count = 0
        for index, line in enumerate(lines):
            if line.startswith("["):
                match = re.search(r"\[(.*?)\]", line)
                if match:
                    content_inside_brackets = match.group(1).strip()
                    index_dict[content_inside_brackets] = count
                    count += 1
    # Pass in a string to var and get its index
    idx_sol = index_dict.get(var)
    # return idx_sol, index_dict
    return idx_sol


def get_group_ndx(
    inp_index_dict: dict[tuple[int, str], Path], lig_resnm: str = "MOL"
) -> dict[str, tuple[int | None, int | None, int | None]]:
    """
    Extract group indices from input index files and organize them into a dictionary.
    This function processes a dictionary of input index files and extracts group indices
    for specific group names such as 'Protein_LIG,' 'Protein,' and 'LIG.'
    The extracted indices are organized into a dictionary where the group names
    are keys, and the corresponding indices are stored as tuples (idx_c, idx_p, idx_l).

    Parameters
    ----------
    inp_index_dict
        A dictionary mapping (replica_num, ligand_name) to index file paths.

    Returns
    ----------
    dict[str, tuple[int, int, int]]
        A dictionary that maps ligand names to group indices as tuples (idx_c, idx_p, idx_l).

    """
    result_dict = {}
    second_element_set = set()

    for key, path in inp_index_dict.items():
        _, second_element = key
        # Check if the second element has already been encountered
        if second_element not in second_element_set:
            result_dict[key] = path
            second_element_set.add(second_element)
    lig_grp_ndx = {}
    for key, val in result_dict.items():
        with open(val, "r") as f:  # val is the file index.ndx
            lines = f.readlines()
            index_dict = {}
            count = 0
            for index, line in enumerate(lines):
                if line.startswith("["):
                    match = re.search(r"\[(.*?)\]", line)
                    if match:
                        content_inside_brackets = match.group(1).strip()
                        index_dict[content_inside_brackets] = count
                    count += 1
        # get index of var
        idx_c = index_dict.get(f"Protein_{lig_resnm}")
        idx_p = index_dict.get("Protein")
        idx_l = index_dict.get(f"{lig_resnm}")
        lig_grp_ndx[key[1]] = (idx_c, idx_p, idx_l)

    return lig_grp_ndx


def merge_pdb(dir_path: Path, input_prot: Path, input_ligand: Path) -> Path:
    """
    Merge protein coordinates and ligand coordinates into one file, complex.pdb

    Parameters
    ----------
    dir_path
        The path where to save the complex.pdb file
    input_prot
        Path to the protein pdb file
    input_ligand
        Path to the protein pdb file

    Returns
    --------
    Path
        A new Path object representing the complex pdb file

    """
    if not dir_path or dir_path == Path():
        raise ValueError("dir_path must be provided and cannot be empty.")

    # Check if the input files exist
    if not input_prot.exists() or not input_ligand.exists():
        raise FileNotFoundError("One or more input files not found.")

    with open(input_prot.as_posix(), "r") as f:
        lines_prot = f.readlines()
        for idx, line in enumerate(lines_prot):
            # protein ff parameters
            if line.startswith("END"):
                print(idx, line)
    # Extract liangad
    atom_lines = []
    with open(input_ligand.as_posix(), "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_lines.append(line.strip())
    # Merge and add END
    merged_list = [l.strip() for l in lines_prot[:idx]] + atom_lines + ["END"]
    # Writeout
    with open(os.path.join(dir_path, "complex.pdb"), "w") as f:
        # Iterate over the list and write each item to the file
        for item in merged_list:
            f.write(item + "\n")
    merge_pdb_output = dir_path / Path("complex.pdb")
    return merge_pdb_output


def merge_top(
    dir_path: Path, input_protein: Path, input_ligand_itp: Path, input_ligand_top: Path
) -> tuple[Path, Path, Path]:
    """
    Create topop.top and .itp files for protein and ligand
    Parameters
    ----------
    dir_path
        the path where to save the output files
    input_protein
        Path to the protein pdb file
    input_ligand_itp
        Path to the ligand itp file
    input_ligand_itp
        Path to the ligand top file

    Returns
    --------
    tuple[Path, Path, Path])
        three output files

    """
    if not dir_path or dir_path == Path():
        raise ValueError("dir_path must be provided and cannot be empty.")

    if not input_protein.exists() or not input_ligand_itp.exists() or not input_ligand_top.exists():
        raise FileNotFoundError("One or more input files not found.")

    with open(input_protein.as_posix(), "r") as f:  # protein.top
        lines = f.readlines()
    fin = []
    vals = [
        "[ moleculetype ]",
        "[ atoms ]",
        "[ bonds ]",
        "[ pairs ]",
        "[ angles ]",
        "[ dihedrals ]",
        "Include Position restraint file",
    ]
    for val in vals:
        # Copy [ moleculetype ] block
        moltype_idx = None
        moltype_idx = [i for i, line in enumerate(lines) if val in line]
        for idx in moltype_idx:
            end_idx_pdb = lines[idx:].index("\n") + idx
            if end_idx_pdb is not None:
                moltype_lines = lines[idx : end_idx_pdb + 1]
                fin += moltype_lines

    # Writeout Protein_chain_B.itp
    prot_itp_name = fin[2].split()[0] + ".itp"
    with open(os.path.join(dir_path, f"{prot_itp_name}"), "w") as f:
        f.writelines(fin)

    out_prot_itp = dir_path / Path(prot_itp_name)  # Output

    # Extract [ atomtypes ] block from the ligand_GMX.itp file
    with open(input_ligand_itp.as_posix(), "r") as f:
        lines = f.readlines()
    start_idx = None
    end_idx = None
    start_idx = lines.index("[ atomtypes ]\n")
    end_idx = lines[start_idx:].index("\n") + start_idx
    atomtypes = lines[start_idx : end_idx + 1]
    lines = lines[:start_idx] + lines[end_idx + 1 :]
    lig_itp_name = input_ligand_itp.as_posix().split("/")[-1]

    # Rewrite the ligand_GMX.itp file
    with open(os.path.join(dir_path, f"{lig_itp_name}"), "w") as f:
        f.writelines(lines)

    out_lig_itp = dir_path / Path(lig_itp_name)  # Output

    # Extract info from ligand_GMX.top
    with open(input_ligand_top.as_posix(), "r") as f:
        lines = f.readlines()
    for line in lines:
        # LIG topology
        if (
            line.startswith("#include")
            and ".itp" in line
            and all([item not in line for item in (".ff", "posre")])
        ):
            ligand_itp = [line]
        # LIG position restraints
        if line.startswith("#ifdef POSRES_"):
            lig_posre_start = lines.index(line)
            lig_posre_end = lines[lig_posre_start:].index("\n") + lig_posre_start
            lig_posre = lines[lig_posre_start : lig_posre_end + 1]
        # LIG name
        if line.startswith("[ molecules ]"):
            lig_name = [lines[lines.index(line) + 2]]

    with open(input_protein.as_posix(), "r") as f:
        lines = f.readlines()
    for line in lines:
        # protein ff parameters
        if line.startswith("#include") and all(item in line for item in (".ff", "forcefield.itp")):
            protein_ff = [line]
        # include water topology
        if line.startswith("#include") and ".ff/tip" in line:
            water_ff = [line]
        # include water ifdef POSRE
        if line.startswith("#ifdef POSRES_"):
            water_posre_start = lines.index(line)
            water_posre_end = lines[water_posre_start:].index("\n") + water_posre_start
            water_posre = lines[water_posre_start : water_posre_end + 1]
        # include topology for ions
        if line.startswith("#include") and "ions.itp" in line:
            ions_line = [line]
        # include system
        if line.startswith("[ system ]"):
            system_start = lines.index(line)
            system_end = lines[system_start:].index("\n") + system_start
            system = lines[system_start : system_end + 1]
        # include molecules
        if line.startswith("[ molecules ]"):
            molecules_start = lines.index(line)
            molecules_end = molecules_start + 2
            molecules = lines[molecules_start : molecules_end + 1]

    # write topol.top file
    lines = (
        protein_ff
        + atomtypes
        + [f'#include "{prot_itp_name}"\n']
        + ligand_itp
        + lig_posre
        + water_ff
        + water_posre
        + ions_line
        + system
        + molecules
        + lig_name
    )
    top_name = Path("topol.top")  # Output
    with open(os.path.join(dir_path, top_name), "w") as f:
        f.writelines(lines)

    out_top = dir_path / Path(top_name)  # Output

    return out_prot_itp, out_lig_itp, out_top


P = TypeVar("P", bound=Path)
T = TypeVar("T")


class SaveFilesFromDict(Node, Generic[T]):
    """
    SaveFilesFromDict class is a Node that receives a dictionary of files and saves them
    to specified locations within a destination folder. The dictionary keys are tuples of
    (int, str), and the values are Path objects.

    """

    inp: Input[dict[tuple[int, str], Path]] = Input(mode="copy")
    """Dictionary of files input (keys are tuples of (int, str), values are Path objects)."""
    destination: FileParameter[Path] = FileParameter(exist_required=False)
    """The destination folder where files will be saved."""
    overwrite: Flag = Flag(default=False)
    """If True, will overwrite any previously existing file in the destination."""

    def run(self) -> None:
        files = self.inp.receive()
        self.logger.info("Files received %s", files)
        files = {k: file.absolute() for k, file in files.items()}

        existence_list = [path.exists() for path in list(files.values())]
        self.logger.info("All files received exist? %s", existence_list)

        dest = self.destination.filepath
        self.logger.info("Parent destination is %s", dest)

        if not dest.is_dir():
            raise ValueError(f"Destination '{dest}' must be a directory")

        files = {k: file.absolute() for k, file in files.items()}

        for k, file in files.items():
            ligand_name = k[1]
            replicanum = k[0]
            if "replica" in file.name:
                mdptp = file.name.split("_")[2]
                dest_path = (
                    dest.absolute() / ligand_name / f"replica{replicanum}" / mdptp / file.name
                )
                self.logger.info(
                    "Ligand_name: %s, Replicanum: %s, MD type: %s", ligand_name, replicanum, mdptp
                )
                self.logger.info("The file recieved will be saved to %s ", dest_path)
            else:
                dest_path = dest.absolute() / ligand_name / file.name
                self.logger.info("Saving file as %s ", dest_path)

            if not dest_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.copyfile(file, dest_path)


@pytest.fixture
def ligand_path(shared_datadir: Any) -> Any:
    return shared_datadir / "other" / "md.log"


@pytest.fixture
def ligand_mv(shared_datadir: Any) -> Any:
    return [
        shared_datadir / "ligs_mv" / "ligandA.pdb",
        shared_datadir / "ligs_mv" / "ligandB.pdb",
    ]


@pytest.fixture
def lig_rep_tp_dirs(shared_datadir: Any) -> Any:
    return [
        shared_datadir / "ligandA" / "Replica1" / "Em",
        shared_datadir / "ligandB" / "Replica1" / "Em",
    ]


def test_add_prefix_to_filename(ligand_path: Any, shared_datadir: Any, test_config: Any) -> None:
    ligand_name = "ligandB"
    replica_num = 1
    mdtp = "Prod"
    # Call the function to get the modified filename
    new_filename = add_prefix_to_filename(ligand_path, ligand_name, replica_num, mdtp)
    # Define the expected result
    expected_result = shared_datadir / "other" / "ligandB_replica1_Prod_md.log"
    # Assert that the result matches the expected value
    assert new_filename == expected_result


def test_extract_filename() -> None:
    data_dict = Path("/test/gromacs/data/other/ligandA_replica1_Prod_md.log")
    ligand_name = "ligandA"
    replica_num = 1
    valid_mdptypes = ["Prod"]
    # Call the function to get the modified filename
    new_filename = extract_filename(data_dict, ligand_name, replica_num, valid_mdptypes)
    # Define the expected result
    expected_result = Path("md.log")
    # Assert that the result matches the expected value
    assert new_filename == expected_result
