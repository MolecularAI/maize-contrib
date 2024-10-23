"""Autodock Vina implementation"""

# pylint: disable=import-outside-toplevel, import-error

from dataclasses import asdict, dataclass
from functools import reduce
from io import StringIO
import itertools
import json
import logging
from pathlib import Path
import re
import shutil
import tarfile
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, cast
from typing_extensions import Self
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, FileParameter, Suffix, Input, Output
from maize.utilities.chem import Isomer, IsomerCollection, smarts_index
from maize.utilities.chem.chem import ChemistryException
from maize.utilities.testing import TestRig
from maize.utilities.validation import SuccessValidator, FileValidator
from maize.utilities.resources import cpu_count
from maize.utilities.utilities import split_multi
from maize.utilities.io import Config

if TYPE_CHECKING:
    from meeko import PDBQTMolecule

AD_HIGH_ENERGY = 1000
SCORE_ONLY_RESULT_REGEX = re.compile(r"\s*Estimated Free Energy of Binding\s*\:\s+(-?\d+\.\d+)\s*")


log = logging.getLogger("run")


def _adv_score_parser_meeko(props: dict[str, str]) -> dict[str, float]:
    """Parse scores from Vina output."""
    prop_name = "free_energy"
    log.debug("Parsing SDF properties '%s'", props)
    value = float(json.loads(props.get("meeko", ""))[prop_name])
    log.debug("Parsed value '%s' from properties", value)
    return {prop_name: value}


PDB_FORMAT = (
    "{:4s}  {:5d} {:4s}{:1s}{:3s} {:1s}{:>4d}{:1s}   "
    "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}       {:4s}{:2s}{:2s}\n"
)


def _pdbqt2pdb(
    pdbqt: Annotated[Path, Suffix("pdbqt")], docked: Annotated[Path, Suffix("pdbqt")], lig_present: bool = True
) -> Annotated[Path, Suffix("pdb")]:
    """Converts PDBQT output to PDB, using flexible receptor information"""
    from meeko import PDBQTReceptor

    rigid = PDBQTReceptor(pdbqt.as_posix()).atoms()
    flex = PDBQTReceptor(docked.as_posix()).atoms()
    inds = flex["serial"]

    # The output contains all concatenated poses, easiest way
    # to distinguish them is on a reset of the index ("serial")
    split_inds = np.array([i + 1 for i, (a, b) in enumerate(itertools.pairwise(inds)) if a > b])
    if lig_present:
        split_inds = split_inds[1::2]
    chunks = np.split(flex, split_inds)

    with docked.with_suffix(".pdb").open("w") as pdb:
        # The odd chunks are the ligand (which is already handled separately
        # and can be ignored), and the even ones are the protein conformations.
        # If lig_present flag is set to true, take all the chunks combined by pairs
        if lig_present:
            chunk_enumerator = enumerate(chunks)
        else:
            chunk_enumerator = enumerate(chunks[1::2])
        for i, chunk in chunk_enumerator:
            chunk["serial"] += rigid["serial"].max()
            full = np.concatenate((rigid, chunk))

            pdb.write(f"MODEL       {i + 1:2d}\n")
            for line in full:
                pdb.write(
                    PDB_FORMAT.format(
                        "ATOM",
                        line["serial"],
                        line["name"],
                        " ",
                        line["resname"],
                        line["chain"],
                        line["resid"],
                        "",
                        line["xyz"][0],
                        line["xyz"][1],
                        line["xyz"][2],
                        line["partial_charges"],
                        0.0,
                        "",
                        line["atom_type"],
                        "",
                    )
                )
            pdb.write("TER\n")
            pdb.write("ENDMDL\n")
    return docked.with_suffix(".pdb")


def _adgpu_score_parser(
    file: Annotated[Path, Suffix("xml")], log: Optional["logging.Logger"] = None
) -> dict[int, dict[str, float | int]]:
    """Parse scores from an AutoDockGPU XML output file"""
    if not file.exists():
        raise FileNotFoundError(f"XML file at '{file.as_posix()}' does not exist")

    tree = ET.parse(file)
    if (res_section := tree.find("result")) is None or (
        rmsd_section := res_section.find("rmsd_table")
    ) is None:
        raise KeyError(f"XML file at '{file.as_posix()}' is malformed or empty")

    results = {}
    for res in rmsd_section:
        if log is not None:
            log.debug(
                "Parsing run '%s' with energy '%s'",
                res.attrib["run"],
                res.attrib["binding_energy"],
            )
        results[int(res.attrib["run"])] = {
            "energy": float(res.attrib["binding_energy"]),
            "cluster_rmsd": float(res.attrib["cluster_rmsd"]),
            "rmsd": float(res.attrib["reference_rmsd"]),
            "cluster": int(res.attrib["rank"]),
        }

    return results


def _list_of_dicts2dict_of_lists(data: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert from a list of dictionaries to a dictionary of lists"""
    return {k: [dic[k] for dic in data] for k in data[0]}


@dataclass
class ADGrid:
    header: dict[str, Path]
    grid_spacing: float
    n_points: NDArray[np.int32]
    n_points_orig: NDArray[np.int32]
    center: NDArray[np.float32]
    grid: NDArray[np.float32]
    coords: NDArray[np.float32]

    @classmethod
    def from_file(cls, file: Path) -> Self:
        """
        Create an AutoDock atom type grid object from a file

        Parameters
        ----------
        file
            Path to the AutoDock grid file

        """
        header: dict[str, Path] = {}
        flat_coords: list[float] = []
        with file.open("r") as inp:
            for line in inp.readlines():
                match line.split():
                    case ["SPACING", x]:
                        grid_spacing = float(x)
                    case [name, data_file]:
                        header[name] = Path(data_file)
                    case ["NELEMENTS", nx, ny, nz]:
                        n_points_orig = np.array([int(nx), int(ny), int(nz)])
                        n_points = n_points_orig + (n_points_orig + 1) % 2
                    case ["CENTER", cx, cy, cz]:
                        center = np.array([float(cx), float(cy), float(cz)])
                    case x:
                        flat_coords.append(float(x[0]))

        mini = center - (n_points / 2) * grid_spacing
        x, y, z = [np.arange(n_points[i]) * grid_spacing + mini[i] for i in range(3)]

        # Match the indexing scheme used by AD4
        grid = np.array(flat_coords).reshape(*n_points[::-1]).T
        coord_grid = np.array(np.meshgrid(z, y, x, indexing="ij")).T

        return cls(
            header=header,
            grid_spacing=grid_spacing,
            n_points=n_points,
            n_points_orig=n_points_orig,
            center=center,
            grid=grid,
            coords=coord_grid,
        )

    def add_bias(
        self, point: tuple[float, float, float], radius: float = 1.2, slope: float = 1e2
    ) -> None:
        """
        Add an energetic bias to the docking grid

        Parameters
        ----------
        point
            Coordinates at which to add the restraint
        radius
            Spherical radius around the restraint point
        slope
            Restraint intensity modulation outside of the radius

        """
        dist = np.sqrt(((self.coords - np.array(point)[::-1]) ** 2).sum(axis=-1))
        self.grid[dist > radius] = dist[dist > radius] * slope

    def to_file(self, out_file: Path) -> None:
        """
        Write the grid out to a file

        Parameters
        ----------
        out_file
            Output file path

        """
        with out_file.open("w") as out:
            out.writelines([f"{key} {path.as_posix()}\n" for key, path in self.header.items()])
            out.write(f"SPACING {self.grid_spacing:4.3f}\n")
            out.write(f"NELEMENTS {' '.join(str(x) for x in self.n_points_orig)}\n")
            out.write(f"CENTER {' '.join(str(x) for x in self.center)}\n")
            out.writelines([f"{val:4.3f}\n" for val in self.grid.T.flatten()])


MeekoAtomParamType = dict[Literal["smarts", "atype", "IDX", "comment"], str | list[int]]


@dataclass
class AtomicConstraint:
    """
    A constraint allowing anchored docking.

    Parameters
    ----------
    smarts
        SMARTS pattern to match in the molecule to dock
    index
        Which atom to constrain in the SMARTS pattern
    location
        Coordinates of the constraint, if not given will use
        the location of a matching atom in the reference ligand
    comment
        An optional comment on the nature of the constraint
    radius
        Radius around the constraint location, larger values are more permissive
    slope
        Restraint intensity outside of the spherical space around the constraint location

    """

    smarts: str
    index: int = 0
    location: tuple[float, float, float] | None = None
    comment: str | None = None
    radius: float = 1.2
    slope: float = 100.0
    counter: int = 0
    _type_spec_abc = "XYZUVWRST"

    _SMARTS_ATOM_MAP = {
        "[#5]": "B",
        "[#7]": "N",
        "[#8]": "O",
        "[#9]": "F",
        "[#15]": "P",
        "[#16]": "S",
        "[#17]": "Cl",
        "[#20]": "Ca",
        "[#35]": "Br",
        "[#53]": "I",
    }

    @property
    def atom_type(self) -> str:
        """The atom type of the SMARTS pattern"""
        return smarts_index(self.smarts, self.index) + self._type_spec_abc[self.counter]

    @property
    def ad_atom_type(self) -> str:
        """The AutoDock atom type"""
        # This is the convenient SMARTS->Type association data
        from meeko.atomtyper import AtomTyper

        params = json.loads(AtomTyper.defaults_json)
        ad_atom_map = [
            {
                "smarts": self._SMARTS_ATOM_MAP.get(entry["smarts"], entry["smarts"]),
                "atype": entry["atype"],
            }
            for entry in params["ATOM_PARAMS"]["alkyl glue"]
        ] + params["ATOM_PARAMS"]["alkyl glue"]

        ad_atom_type: str | None = None
        query = Chem.MolFromSmarts(self.smarts)
        for entry in ad_atom_map:
            atype_query = Chem.MolFromSmarts(entry["smarts"])

            # Because we're matching a SMARTS with a SMARTS, this can fail for
            # various reasons (probably because we're not matching a real molecule)
            try:
                # query-query matches are important, otherwise
                # SMARTS specifying multiple substructures won't match
                matched = [
                    *query.GetSubstructMatch(atype_query, useQueryQueryMatches=True),
                    *query.GetSubstructMatch(atype_query),
                ]
            except RuntimeError:
                continue

            if matched and self.index in matched:
                ad_atom_type = entry["atype"]

        if ad_atom_type is None:
            raise ChemistryException(f"No valid AD atom type found for '{self.smarts}'")
        return ad_atom_type

    def inc_atom_type(self) -> None:
        """Increment the atom type number"""
        self.counter += 1

    def to_meeko(self) -> MeekoAtomParamType:
        """Create a parameter entry for Meeko"""
        data: MeekoAtomParamType = {
            "smarts": self.smarts,
            "atype": self.atom_type,
            "IDX": [self.index + 1],
        }
        if self.comment is not None:
            data["comment"] = self.comment
        return data


def _update_map(file: Path, atom_type: str) -> None:
    """Update the main AD map with a new atom type"""

    with file.open("r") as inp:
        header = []
        variables = []
        labels = []
        coords = []
        other = {}
        for line in inp.readlines():
            tokens = split_multi(line, " \t=")
            no_comments: list[str] = []
            for token in tokens:
                if token == "#":
                    break
                no_comments.append(token)

            match no_comments:
                case ["variable", idx, "file", map_file, "filetype", filetype, "skip", skip]:
                    variables.append((int(idx), map_file, filetype, int(skip)))
                case ["label", val]:
                    labels.append(val)
                case ["coord", idx, "file", map_file, "filetype", filetype, "offset", offset]:
                    coords.append((int(idx), map_file, filetype, int(offset)))

                # For some bizarre reason, certain comments are required in maps.fld files
                case [comment_key, *vals] if comment_key.startswith("#"):
                    header.append((comment_key.strip(), *(v.strip() for v in vals)))
                case [key, val, *_]:
                    other[key] = val

    # Add our custom atom type files, not sure if it has to be the 3rd-to-last
    # position in both lists, but the reference implementation does this
    base_name = file.stem.rstrip(file.suffixes[0])
    labels.insert(-2, f"{atom_type}-affinity")
    prev_idx = variables[-3][0]
    variables.insert(-2, (prev_idx + 1, f"{base_name}.{atom_type}.map", "ascii", 6))
    for i, var in enumerate(variables[-2:]):
        variables[prev_idx + i + 1] = (var[0] + 1, *var[1:])

    with file.open("w") as out:
        out.writelines(f"{key} {' '.join(vals)}\n" for key, *vals in header)
        out.writelines(f"{key}={val}\n" for key, val in other.items())
        out.writelines(
            f"coord {idx} file={file} filetype={ft} offset={os}\n" for idx, file, ft, os in coords
        )
        out.writelines(f"label={val}\n" for val in labels)
        out.writelines(
            f"variable {idx} file={file} filetype={ft} skip={skip}\n"
            for idx, file, ft, skip in variables
        )


# Helper functions using meeko to convert to PDBQT...
def _mol2pdbqt(
    file: Path, isomer: "Isomer", constraints: list[AtomicConstraint] | None = None
) -> None:
    """Converts an isomer to a PDBQT file using meeko"""
    from meeko import MoleculePreparation

    if constraints:
        from meeko.atomtyper import AtomTyper

        param = json.loads(AtomTyper.defaults_json)
        param["ATOM_PARAMS"]["maize"] = [cons.to_meeko() for cons in constraints]
        preparator = MoleculePreparation(atom_type_smarts=param)
    else:
        preparator = MoleculePreparation()

    preparator.prepare(isomer._molecule)
    preparator.write_pdbqt_file(file)


# ...and from DLG (a kind of PDBQT file) to SDF
def _adv2sdf(inp_file: Path, sdf: Path) -> None:
    """Converts an AD DLG file to SDF using meeko"""
    from meeko import PDBQTMolecule

    with inp_file.open() as inp:
        mol = PDBQTMolecule(inp.read(), is_dlg=inp_file.suffix == ".dlg", skip_typing=True)
    with sdf.open("w") as sdfout:
        out, failures = _write_sd_string(mol)
        sdfout.write(out)
    if len(failures) > 0:
        raise IOError(f"Meeko failed to write file '{sdf.as_posix()}'")


# This is a temporary patched function originally from meeko
# TODO: Open a PR for this upstream
def _write_sd_string(
    pdbqt_mol: "PDBQTMolecule", only_cluster_leads: bool = False
) -> tuple[str, list[int]]:
    """Creates an SDF string from a meeko PDBQTMolecule"""
    from meeko import RDKitMolCreate

    sio = StringIO()
    f = Chem.SDWriter(sio)
    mol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol, only_cluster_leads)
    failures = [i for i, mol in enumerate(mol_list) if mol is None]
    combined_mol = RDKitMolCreate.combine_rdkit_mols(mol_list)
    if combined_mol is None:
        return "", failures
    nr_conformers = combined_mol.GetNumConformers()
    property_names = {
        "free_energy": "free_energies",
        "intermolecular_energy": "intermolecular_energies",
        "internal_energy": "internal_energies",
        "cluster_size": "cluster_size",
        "cluster_id": "cluster_id",
        "rank_in_cluster": "rank_in_cluster",
    }
    props = {}
    if only_cluster_leads:
        nr_poses = len(pdbqt_mol._pose_data["cluster_leads_sorted"])
        pose_idxs = pdbqt_mol._pose_data["cluster_leads_sorted"]
    else:
        nr_poses = pdbqt_mol._pose_data["n_poses"]
        pose_idxs = list(range(nr_poses))
    for prop_sdf, prop_pdbqt in property_names.items():
        if nr_conformers == nr_poses:
            props[prop_sdf] = prop_pdbqt

    # The original checked if all keys were set for all conformers and would fail if
    # any of them weren't set. This caused problems for versions of Vina that do not
    # write out cluster information, as those properties are empty, and thus meeko
    # would refuse to write even the existing properties. Here we just check each
    # property individually, and write it if it's available.
    available_props = {}
    for orig_key, key in props.items():
        if len(pdbqt_mol._pose_data[key]) == nr_conformers:
            available_props[orig_key] = key
    for conformer in combined_mol.GetConformers():
        i = conformer.GetId()
        j = pose_idxs[i]
        data = {k: pdbqt_mol._pose_data[v][j] for k, v in available_props.items()}
        if len(data):
            combined_mol.SetProp("meeko", json.dumps(data))
        f.write(combined_mol, i)
    f.close()
    output_string = sio.getvalue()
    return output_string, failures


class PreparePDBQT(Node):
    """Prepares a receptor for docking with Vina."""

    tags = {"chemistry", "docking", "preparation"}

    required_callables = ["prepare_receptor", "prepare_flex_receptor"]
    """
    Requires various scripts and tools:

    prepare_receptor
        Included in ``AutoDockTools``.

    """
    _RepairType = Literal["bonds_hydrogens", "bonds", "hydrogens", "checkhydrogens", "None"]
    _CleanupType = Literal["nphs", "lps", "waters", "nonstdres", "deleteAltB"]

    inp: Input[Annotated[Path, Suffix(".pdb")]] = Input()
    """Receptor structure without ligand"""

    out: Output[Annotated[Path, Suffix(".pdbqt")]] = Output()
    """Tar archive of all grid files"""

    out_flex: Output[Annotated[Path, Suffix(".pdbqt")]] = Output(optional=True)
    """Flexible receptor output"""

    repairs: Parameter[_RepairType] = Parameter(default="None")
    """Types of repairs to be done to the PDB file"""

    preserve_charges: Flag = Flag(default=False)
    """Whether to preserve existing charges instead of adding Gasteiger charges"""

    cleanup_protein: Parameter[list[_CleanupType]] = Parameter(
        default_factory=lambda: ["nphs", "lps", "waters", "nonstdres"]
    )
    """Cleanup options"""

    remove_nonstd: Flag = Flag(default=False)
    """Remove non-standard residues"""

    flexible_residues: Parameter[list[str]] = Parameter(default_factory=list)
    """List of flexible receptors, in the format XXX123"""

    def run(self) -> None:
        structure = self.inp.receive()

        receptor_pdbqt = Path("rec.pdbqt")
        command = (
            f"{self.runnable['prepare_receptor']} "
            f"-A '{self.repairs.value}' "
            f"-U '{'_'.join(self.cleanup_protein.value)}' "
            f"-r {structure.as_posix()} "
            f"-o {receptor_pdbqt.as_posix()} "
        )
        if self.preserve_charges.value:
            command += "-C "
        if self.remove_nonstd.value:
            command += "-e "

        self.run_command(command, validators=[FileValidator(receptor_pdbqt)], verbose=True)
        if self.flexible_residues.value:
            flex_output = Path("rec_flex.pdbqt")
            rigid_output = Path("rec_rigid.pdbqt")
            residues = ",".join(self.flexible_residues.value)
            command = (
                f"{self.runnable['prepare_flex_receptor']} "
                f"-r {receptor_pdbqt.as_posix()} -s {residues}"
            )
            self.run_command(command, validators=[FileValidator([rigid_output, flex_output])])
            self.out.send(rigid_output)
            self.out_flex.send(flex_output)
            return

        self.out.send(receptor_pdbqt)


class PrepareGrid(Node):
    """Prepares a receptor for docking with AutoDock4."""

    tags = {"chemistry", "docking", "preparation"}

    required_callables = ["prepare_receptor", "write_gpf", "autogrid"]
    """
    Requires various scripts and tools:

    write_gpf
        Script to create GPF output with all possible atomtypes,
        `from here <https://github.com/diogomart/write-autogrid-config>`_.

    prepare_receptor
        Included in ``AutoDockTools``.

    autogrid
        Included in the normal CPU-only version of AutoDock

    """

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko`` installed"""

    inp_structure: Input[Annotated[Path, Suffix(".pdb", ".pdbqt")]] = Input()
    """Receptor structure without ligand"""

    inp_ligand: Input[Isomer] = Input(optional=True)
    """Reference ligand structure, if not provided requires `search_center` to be set"""

    out: Output[Annotated[Path, Suffix("tar")]] = Output()
    """Tar archive of all grid files"""

    search_center: Parameter[tuple[float, float, float]] = Parameter(
        default=(np.nan, np.nan, np.nan)
    )
    """Center of the search space for docking, required if `inp_ligand` is not given"""

    search_range: Parameter[tuple[float, float, float]] = Parameter(default=(15.0, 15.0, 15.0))
    """Range of the search space for docking"""

    constraints: Parameter[list[AtomicConstraint]] = Parameter(default_factory=list)
    """
    List of atomic position constraints to add to the grid. For the general idea, see the
    `anchored docking tutorial <https://github.com/ccsb-scripps/AutoDock-GPU/wiki/Anchored-docking>`_.
    
    """

    def run(self) -> None:
        structure = self.inp_structure.receive()

        # Create receptor PDBQT (if needed)
        receptor_pdbqt = Path("rec.pdbqt")
        if structure.suffix == ".pdbqt":
            shutil.move(structure, receptor_pdbqt)
        else:
            self.logger.info("Preparing receptor '%s'", structure.name)
            self.run_command(
                f"{self.runnable['prepare_receptor']} "
                f"-r {structure.as_posix()} "
                f"-o {receptor_pdbqt.as_posix()}",
                validators=[FileValidator(receptor_pdbqt)],
            )

        constraints = self.constraints.value

        # Create temporary ligand PDBQT
        if self.inp_ligand.ready():
            lig = self.inp_ligand.receive()
            self.logger.info("Using reference ligand '%s'", lig.name or lig.inchi)

            # Update the constraint with reference locations if needed
            for constraint in constraints:
                if constraint.location is None:
                    matches = lig.find_match(constraint.smarts)

                    # Multiple matches will cause problems during docking later, as one
                    # atom would have to be in multiple locations at the same time...
                    if len(matches) != 1:
                        raise ChemistryException(
                            f"The specified constraint with SMARTS '{constraint.smarts}' does not"
                            f" have exactly one match for ligand '{lig.to_smiles(remove_h=True)}'"
                        )

                    # We should only have one match, and we use the user-provided
                    # index in the SMARTS to pick the correct atom, followed by
                    # indexing in to the first conformer to get its position
                    index_entry = matches[0][constraint.index]
                    constraint.location = cast(
                        tuple[float, float, float], tuple(lig.coordinates[0][index_entry])
                    )
                    self.logger.info(
                        "Placing constraint for '%s' at %s based on reference",
                        constraint.smarts,
                        constraint.location,
                    )

            lig_pdbqt = Path("lig.pdbqt")
            _mol2pdbqt(lig_pdbqt, lig)
            command = (
                f"{self.runnable['write_gpf']} "
                f"-p {max(self.search_range.value):5.3f} "
                f"-l {lig_pdbqt.as_posix()} "
                f"{receptor_pdbqt.as_posix()}"
            )
        else:
            assert all(np.isfinite(c) for c in self.search_center.value)
            box_config = Path("box.txt")
            with box_config.open("w") as conf:
                for axis, coord, size in zip(
                    ("x", "y", "z"), self.search_center.value, self.search_range.value
                ):
                    conf.write(f"center_{axis} = {coord:5.3f}\n")
                    conf.write(f"size_{axis} = {size:5.3f}\n")
            command = (
                f"{self.runnable['write_gpf']} "
                f"-b {box_config.as_posix()} "
                f"{receptor_pdbqt.as_posix()}"
            )

        # Create GPF, includes search geometry and index of needed maps
        gpf = receptor_pdbqt.with_suffix(".gpf")
        self.run_command(command, validators=[FileValidator(gpf)])

        # Create maps
        self.logger.info("Building all maps")
        glg = gpf.with_suffix(".glg")
        fld = glg.with_suffix(".maps.fld")
        self.run_command(
            f"{self.runnable['autogrid']} -p {gpf.as_posix()} -l {glg.as_posix()}",
            validators=[FileValidator(glg), FileValidator(fld)],
        )

        # Add constraint maps
        for constraint in constraints:
            if constraint.location is None:
                raise ChemistryException(
                    f"Constraint '{constraint.smarts}' does not have a "
                    "location, and no reference ligand was provided"
                )

            base_file = next(Path().glob(f"*.{constraint.ad_atom_type}.map"))
            self.logger.info(
                "Using '%s' as a basis for constraint on '%s'", base_file, constraint.smarts
            )
            grid = ADGrid.from_file(base_file)
            grid.add_bias(constraint.location, radius=constraint.radius, slope=constraint.slope)

            # Make sure we're not overwriting constraint maps, we can
            # have multiple ones for the same element, e.g. NX0, NX1, ...
            new_file = Path(f"rec.{constraint.atom_type}.map")
            while new_file.exists():
                constraint.inc_atom_type()
                new_file = Path(f"rec.{constraint.atom_type}.map")

            grid.to_file(new_file)
            _update_map(fld, constraint.atom_type)
            self.logger.info(
                "Created new map '%s' for constraint '%s'", new_file, constraint.smarts
            )

        cons_file = Path("constraints.json")
        if constraints:
            with cons_file.open("w") as out:
                json.dump([asdict(con) for con in constraints], out)

        # Wrap it all up
        tar = Path("grid.tar")
        with tarfile.open(tar, "w") as archive:
            for file in Path().glob("*.map"):
                archive.add(file)
            for file in (receptor_pdbqt, gpf, glg, fld):
                archive.add(file)
            if cons_file.exists():
                archive.add(cons_file)
        self.out.send(tar)


class AutoDockGPU(Node):
    """
    Runs AutoDock on the GPU.

    See [#santos2021]_ for scientific details.

    Notes
    -----
    Clone the repo from `here <https://github.com/ccsb-scripps/AutoDock-GPU>`_,
    load modules for the compiler and CUDA, set ``GPU_INCLUDE_PATH`` and
    ``GPU_LIBRARY_PATH``, and run ``make DEVICE=CUDA``. This also requires
    `meeko <https://github.com/forlilab/Meeko>`_ to convert to and from pdbqt
    files, specify `mk_prepare` and `mk_export`.

    If you get very high docking scores this often means that the ligand is outside
    of the grid. This can be due to a map that is too small (increase ``search_range``)
    or a misplaced box that is hard to access (modify ``search_center``).

    References
    ----------
    .. [#santos2021] Santos-Martins, D. et al. Accelerating AutoDock4 with GPUs
       and Gradient-Based Local Search. J. Chem. Theory Comput. 17, 1060-1073 (2021).

    """
    tags = {"chemistry", "docking", "scorer", "tagger"}

    SCORE_TAG = "energy"

    required_callables = ["autodock_gpu"]
    """Requires the ``autodock_gpu`` executable"""

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko==0.4`` installed"""

    inp: Input[list[IsomerCollection]] = Input()
    """
    List of molecules to dock, each molecule can have multiple isomers,
    these will be docked separately.

    """

    inp_grid: Input[Annotated[Path, Suffix("tar")]] = Input(cached=True, optional=True)
    """The protein grid file, all internally referenced files must be available"""

    out: Output[list[IsomerCollection]] = Output(optional=True)
    """
    Docked molecules with conformations and scores attached. Also
    include per-conformer clustering information performed by
    AutoDock, use the keys 'rmsd', 'cluster_rmsd', 'cluster' to access.

    """

    out_scores: Output[NDArray[np.float32]] = Output()
    """Docking scores, the best for each docked IsomerCollection"""

    grid_file: FileParameter[Annotated[Path, Suffix("fld")]] = FileParameter(optional=True)
    """The protein grid file, all internally referenced files must be available"""

    ref_ligand: Parameter[Isomer] = Parameter(optional=True)
    """Optional reference ligand for RMSD analysis"""

    seed: Parameter[int] = Parameter(default=42)
    """The default seed"""

    heuristics: Parameter[int] = Parameter(default=1)
    """Number of evaluations for ligand-based automatic search"""

    heurmax: Parameter[int] = Parameter(default=12000000)
    """Heuristics evaluation limit"""

    nrun: Parameter[int] = Parameter(default=20)
    """LGA runs"""

    population_size: Parameter[int] = Parameter(default=150)
    """LGA population size"""

    lsit: Parameter[int] = Parameter(default=300)
    """Local search iterations"""

    derivtypes: Parameter[dict[str, str]] = Parameter(default_factory=dict)
    """Atomtype mappings to add to ``derivtype``, e.g. NA->N"""

    strict: Flag = Flag(default=False)
    """When set, raises an exception if docking a molecule failed, otherwise logs a warning"""

    scores_only: Flag = Flag(default=False)
    """If ``True``, will only return the scores and no conformers"""

    constraints: Flag = Flag(default=True)
    """Use constraints if they are included in the grid"""

    def run(self) -> None:
        mols = self.inp.receive()

        # Molecule inputs
        inputs = Path("inputs")
        inputs.mkdir()

        constraints = []

        # Grid
        if self.inp_grid.is_set and (tar := self.inp_grid.receive_optional()) is not None:
            grid = Path("grid")
            grid.mkdir()
            with tarfile.open(tar, "r") as archive:
                archive.extractall(grid)
            grid_file = next(grid.glob("*.maps.fld"))
            cons_file = grid / "constraints.json"
            if cons_file.exists():
                with cons_file.open("r") as con:
                    constraints = [AtomicConstraint(**data) for data in json.load(con)]
        else:
            grid_file = self.grid_file.filepath

        # Convert all ligands to pdbqt and collect
        # their paths and names in a batch file
        batch_file = Path("batch.txt")
        with batch_file.open("w") as file:
            file.write(f"{grid_file.as_posix()}\n")
            for mol in mols:
                for isomer in mol.molecules:
                    # Each constraint should have a maximum of one match, as multiple matches
                    # will not be able to satisfy the grid placement. No matches will cause the
                    # special constraint atomtype to have no effect.
                    if (
                        any(len(isomer.find_match(con.smarts)) > 1 for con in constraints)
                        and self.constraints.value
                    ):
                        self.logger.warning(
                            "Isomer '%s' ('%s') does not match constraints '%s'",
                            isomer.name or isomer.inchi,
                            isomer.to_smiles(remove_h=True),
                            [con.smarts for con in constraints],
                        )
                        continue

                    # Tools like REINVENT rely on getting the same number of scores out
                    # as molecules, so we can't filter out failed embeddings earlier...
                    if isomer.n_conformers == 0:
                        self.logger.warning(
                            "No embedding for '%s' ('%s'), skipping...",
                            isomer.name or isomer.inchi,
                            isomer.to_smiles(remove_h=True),
                        )
                        continue

                    ligand = inputs / f"{isomer.inchi}.pdbqt"

                    # Create pdbqt input
                    try:
                        _mol2pdbqt(
                            ligand,
                            isomer,
                            constraints=constraints if self.constraints.value else None,
                        )
                    except RuntimeError as err:
                        self.logger.warning(
                            "Failed to convert '%s' ('%s'), skipping...",
                            isomer.name or isomer.inchi,
                            isomer.to_smiles(remove_h=True),
                            exc_info=err,
                        )
                        continue

                    file.write(f"{ligand.absolute().as_posix()}\n")
                    file.write(f"{isomer.inchi}\n")

        command = (
            f"{self.runnable['autodock_gpu']} --filelist {batch_file.as_posix()} "
            f"--heuristics {self.heuristics.value} --nrun {self.nrun.value} "
            f"--psize {self.population_size.value} --lsit {self.lsit.value} "
            f"--seed {self.seed.value} --heurmax {self.heurmax.value}"
        )

        # Possible reference ligand
        if self.ref_ligand.is_set:
            ref_ligand = Path("ref_ligand.pdbqt")
            _mol2pdbqt(ref_ligand, self.ref_ligand.value)
            command += f" --xraylfile {ref_ligand.as_posix()}"

        # Possible manually specified derivtypes
        derivtypes = []
        if self.derivtypes.value:
            derivtypes.extend([f"{key}={value}" for key, value in self.derivtypes.value.items()])

        # When using constraints, we may need to refer to the original AD atom types
        if self.constraints.value:
            derivtypes.extend([f"{con.atom_type}={con.ad_atom_type}" for con in constraints])

        if derivtypes:
            command += f" --derivtype {'/'.join(derivtypes)}"

        validators = [SuccessValidator("All jobs ran without errors")] if self.strict.value else []
        with self.gpus(1):
            self.run_command(command, verbose=True, validators=validators)

        # Collect outputs
        for mol in mols:
            for isomer in mol.molecules:
                output = Path(f"{isomer.inchi}.xml")

                # Failed dockings and missing embeddings get a NaN
                try:
                    # Isomer scores are in the order of the indices
                    results = _adgpu_score_parser(output, log=self.logger)
                except (KeyError, FileNotFoundError) as err:
                    if self.strict.value:
                        raise err
                    self.logger.warning("Docking isomer '%s' failed", isomer.name or isomer.inchi)
                    isomer.add_score(self.SCORE_TAG, np.full(self.nrun.value, np.nan))
                    continue

                res_transpose = _list_of_dicts2dict_of_lists(list(results.values()))

                # High energies indicate grid problems
                if any(ener > AD_HIGH_ENERGY for ener in res_transpose["energy"]):
                    self.logger.warning(
                        "Isomer '%s' ('%s') has runs with high energy poses. This indicates "
                        "a possible lack of grid coverage or a poorly-defined search space. "
                        "Adjust `search_range` and `search_center` during grid preparation.",
                        isomer.name or isomer.inchi,
                        mol.smiles,
                    )

                # Convenience score attribute
                isomer.set_tag("score_type", "oracle")
                isomer.set_tag("origin", self.name)
                isomer.add_score(self.SCORE_TAG, np.array(res_transpose["energy"]))
                self.logger.info(
                    "Parsed isomer '%s', score %s",
                    isomer.name or isomer.inchi,
                    isomer.scores[self.SCORE_TAG],
                )

                # We only parse the conformers if the user asks for them,
                # otherwise it unnecessarily slows things like REINVENT down
                if not self.scores_only.value:
                    # This allows us to convert all pdbqt outputs
                    # into one SDF, with the scoring order
                    sdf_out = Path(f"{isomer.inchi}-out.sdf")
                    try:
                        _adv2sdf(output.with_suffix(".dlg"), sdf_out)
                    except RuntimeError as err:
                        self.logger.warning(
                            "Meeko failed to convert %s, skipping conformer parsing",
                            isomer.name or isomer.inchi,
                            exc_info=err,
                        )
                        continue

                    # Add all conformers and set their coords
                    isomer.update_conformers_from_sdf(sdf_out)

                    # AD gives us lots of useful information for each conformer, e.g. energy
                    # reference RMSD and cluster, we tag each conformer with this information
                    for key, vals in res_transpose.items():
                        for conf, val in zip(isomer.conformers, vals):
                            conf.set_tag(key, val)

            mol.primary_score_tag = self.SCORE_TAG

        if not self.scores_only.value:
            self.out.send(mols)
        self.out_scores.send(np.array([mol.primary_score for mol in mols]))


class _Vina(Node, register=False):
    """Base for all Vina variants"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to dock"""

    out: Output[list[IsomerCollection]] = Output()
    """Docked molecules with conformations and scores attached"""

    seed: Parameter[int] = Parameter(default=42)
    """The default seed"""

    n_jobs: Parameter[int] = Parameter(default=cpu_count())
    """Number of docking runs to perform in parallel"""

    n_poses: Parameter[int] = Parameter(default=1)
    """Number of poses to generate"""

    receptor: FileParameter[Annotated[Path, Suffix("pdbqt")]] = FileParameter()
    """Path to the receptor structure"""

    search_center: Parameter[tuple[float, float, float]] = Parameter()
    """Center of the search space for docking"""

    search_range: Parameter[tuple[float, float, float]] = Parameter(default=(15.0, 15.0, 15.0))
    """Range of the search space for docking"""

    sanitize: Flag = Flag(default=True)
    """
    Whether to sanitize the parsed output from ADV. There may be rare cases in
    which the default RDKit sanitization causes issues, in this case set this
    flag to ``False`` and perform your own sanitization later, or accept
    potentially scrambled bond information.

    """

    def _parse_adv_outputs(
        self, mols: list[IsomerCollection], mol_outputs: list[list[Path]]
    ) -> None:
        """Parses ADV output, including conformers and scores from PDBQT or DLG outputs"""
        moldict = {iso.inchi: iso for mol in mols for iso in mol.molecules}
        outdict = {file.stem.strip("_out"): file for folder in mol_outputs for file in folder}
        for i, (key, file) in enumerate(outdict.items()):
            isomer = moldict[key]
            self.logger.info("Parsing isomer %s: '%s'", i, isomer.name or isomer.inchi)
            if not file.exists() or file.stat().st_size == 0:
                self.logger.warning(
                    "Docking failed for '%s' (%s)", isomer.name or isomer.inchi, isomer
                )
                continue
            _adv2sdf(file, file.with_suffix(".sdf"))
            isomer.update_conformers_from_sdf(
                file.with_suffix(".sdf"), score_parser=_adv_score_parser_meeko, sanitize=self.sanitize.value
            )
            isomer.set_tag("origin", self.name)


class Vina(_Vina):
    """
    Runs Vina on a molecule input.

    See [#eberhardt2021]_ and [#trott2010]_.

    The step expects to either find a ``vina`` executable in the ``PATH``,
    an appropriate module defined in ``config.toml``, or a module specified
    using the :attr:`~maize.core.node.Node.modules` attribute.

    References
    ----------
    .. [#eberhardt2021] Eberhardt, J., Santos-Martins, D., Tillack, A. F. & Forli, S.
       AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field,
       and Python Bindings. J. Chem. Inf. Model. 61, 3891-3898 (2021).

    .. [#trott2010] Trott, O. & Olson, A. J. AutoDock Vina: Improving the speed and accuracy
       of docking with a new scoring function, efficient optimization, and
       multithreading. Journal of Computational Chemistry 31, 455-461 (2010).

    """
    tags = {"chemistry", "docking", "scorer", "tagger"}

    required_callables = ["vina"]
    """Requires the ``vina`` executable"""

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko==0.4`` installed"""

    exhaustiveness: Parameter[int] = Parameter(default=8)
    """Exhaustiveness of the global search (roughly proportional to time)"""

    def run(self) -> None:
        def _skip_msg(isomer: Isomer, err: Exception | str) -> None:
            self.logger.warning(
                "Skipping '%s' due to PDBQT conversion error:\n %s",
                isomer.name or isomer.inchi,
                err,
            )

        mols = self.inp.receive()

        x, y, z = self.search_center.value
        dx, dy, dz = self.search_range.value

        # Collect all docking commands to be executed, create directories
        lig_temp = Path("lig-temp.pdbqt")
        commands: list[str] = []
        mol_outputs: list[list[Path]] = []
        for i, mol in enumerate(mols):
            mol_path = Path(f"mol-{i}")
            mol_path.mkdir()
            isomer_outputs: list[Path] = []
            self.logger.info("Docking molecule %s: '%s'", i, mol)
            for j, isomer in enumerate(mol.molecules):
                self.logger.debug("  Docking isomer %s: '%s'", j, isomer.name or isomer.inchi)
                iso_path = mol_path / f"isomer-{j}"
                iso_path.mkdir()
                ligand = iso_path / "input.pdbqt"
                docked = iso_path / f"{isomer.inchi}_out.pdbqt"
                try:
                    _mol2pdbqt(lig_temp, isomer)
                    _clean_pdbqt_atomtypes(lig_temp, ligand)
                except ValueError as err:
                    _skip_msg(isomer, err)

                if not ligand.exists():
                    _skip_msg(isomer, "File not found")

                command = (
                    f"{self.runnable['vina']} --receptor {self.receptor.filepath.as_posix()} "
                    f"--ligand {ligand.as_posix()} "
                    f"--cpu 1 --seed {self.seed.value} --out {docked.as_posix()} "
                    f"--num_modes {self.n_poses.value} "
                    f"--exhaustiveness {self.exhaustiveness.value} "
                    f"--center_x {x} --center_y {y} --center_z {z} "
                    f"--size_x {dx} --size_y {dy} --size_z {dz} "
                )
                commands.append(command)
                isomer_outputs.append(docked)
            mol_outputs.append(isomer_outputs)

        # Run all commands at once
        self.run_multi(
            commands,
            verbose=True,
            raise_on_failure=False,
            n_jobs=self.n_jobs.value,
        )

        # Convert each pose to SDF, update isomer conformation
        self._parse_adv_outputs(mols, mol_outputs)
        self.out.send(mols)


class VinaFlex(_Vina):
    """
    Runs Vina on a molecule input.

    The step expects to either find a ``vina`` executable in the ``PATH``,
    an appropriate module defined in ``config.toml``, or a module specified
    using the :attr:`~maize.core.node.Node.modules` attribute.

    """
    tags = {"chemistry", "docking", "scorer", "tagger"}

    required_callables = ["vina"]
    """Requires the ``vina`` executable"""

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko==0.4`` installed"""

    out_residues: Output[list[Path]] = Output()
    """
    Flexible residue output, a dictionary of SDF files containing
    flexible residue coordinates for each molecule INCHI.

    """

    out_complex: Output[list[Path]] = Output()
    """Complex SDF output"""

    exhaustiveness: Parameter[int] = Parameter(default=8)
    """Exhaustiveness of the global search (roughly proportional to time)"""

    flex_receptor: FileParameter[Annotated[Path, Suffix("pdbqt")]] = FileParameter()
    """
    Receptor file for flexible docking
    
    `Vina flexible docking tutorial <https://autodock-vina.readthedocs.io/en/latest/docking_flexible.html>`_

    """

    n_cores_per_job: Parameter[int] = Parameter(default=1)
    """Number of cores to use per job"""

    lig_present: Flag = Flag(default=True)
    """Flag to specify whether ligand is present in the docked PDB complex output"""

    @staticmethod
    def _split_mol_sdf(sdf: Path) -> tuple[Path, Path]:
        """Splits a molecule SDF into separate files depending on fragments"""
        base = sdf.stem
        ligand_poses = []
        res_poses = []
        with Chem.SDMolSupplier(sdf.as_posix(), removeHs=False) as supp:
            for i, mol in enumerate(supp):
                if mol is None:
                    continue
                # Properties are lost on fragmentation
                props = mol.GetPropsAsDict()

                # First fragment is (hopefully always?) the ligand
                lig, *res = Chem.GetMolFrags(mol, asMols=True)

                # Properties are lost on fragmentation, so we reset them to the parent
                for name, prop in props.items():
                    lig.SetProp(name, prop)

                ligand_poses.append(lig)
                residues = reduce(Chem.CombineMols, res)
                res_poses.append(residues)

        lig_path = sdf.parent / f"{base}-lig.sdf"
        with Chem.SDWriter(lig_path.as_posix()) as writer:
            for pose in ligand_poses:
                writer.write(pose)

        # All the other fragments are the interacting, flexible residues
        res_path = sdf.parent / f"{base}-res.sdf"
        with Chem.SDWriter(res_path.as_posix()) as writer:
            for pose in res_poses:
                writer.write(pose)

        return lig_path, res_path

    def _parse_flex_adv_outputs(
        self,
        mols: list[IsomerCollection],
        mol_outputs: list[list[Path]],
        receptor: Path,
    ) -> tuple[dict[str, Path], dict[str, Path]]:
        """Parses ADV output, including conformers and scores from PDBQT or DLG outputs"""
        moldict = {iso.inchi: iso for mol in mols for iso in mol.molecules}
        outdict = {file.stem.strip("_out"): file for folder in mol_outputs for file in folder}
        resout = {}
        complexes = {}
        for i, (key, file) in enumerate(outdict.items()):
            isomer = moldict[key]
            self.logger.info("Parsing isomer %s: '%s'", i, isomer)
            if not file.exists() or file.stat().st_size == 0:
                self.logger.warning(
                    "Docking failed for '%s' (%s)", isomer.name or isomer.inchi, isomer
                )
                continue
            _adv2sdf(file, file.with_suffix(".sdf"))
            complexes[key] = file.with_suffix(".sdf")
            lig, _ = self._split_mol_sdf(file.with_suffix(".sdf"))
            resout[key] = _pdbqt2pdb(receptor, docked=file, lig_present=self.lig_present.value)
            isomer.update_conformers_from_sdf(
                lig, score_parser=_adv_score_parser_meeko, sanitize=False
            )
            isomer.set_tag("origin", self.name)
        return resout, complexes

    def run(self) -> None:
        mols = self.inp.receive()

        x, y, z = self.search_center.value
        dx, dy, dz = self.search_range.value

        # Collect all docking commands to be executed, create directories
        lig_temp = Path("lig-temp.pdbqt")
        commands: list[str] = []
        mol_outputs: list[list[Path]] = []
        for i, mol in enumerate(mols):
            mol_path = Path(f"mol-{i}")
            mol_path.mkdir()
            isomer_outputs: list[Path] = []
            self.logger.info("Docking molecule %s: '%s'", i, mol)
            for j, isomer in enumerate(mol.molecules):
                self.logger.debug("  Docking isomer %s: '%s'", j, isomer)
                iso_path = mol_path / f"isomer-{j}"
                iso_path.mkdir()
                ligand = iso_path / "input.pdbqt"
                docked = iso_path / f"{isomer.inchi}_out.pdbqt"
                try:
                    _mol2pdbqt(lig_temp, isomer)
                    _clean_pdbqt_atomtypes(lig_temp, ligand)
                except ValueError as err:
                    self.logger.warning(
                        "Skipping '%s' due to PDBQT conversion error:\n %s",
                        isomer.name or isomer.inchi,
                        err,
                    )

                command = (
                    f"{self.runnable['vina']} --receptor {self.receptor.filepath.as_posix()} "
                    f"--ligand {ligand.as_posix()} "
                    f"--cpu {self.n_cores_per_job.value} --seed {self.seed.value} "
                    f"--out {docked.as_posix()} "
                    f"--num_modes {self.n_poses.value} "
                    f"--exhaustiveness {self.exhaustiveness.value} "
                    f"--center_x {x} --center_y {y} --center_z {z} "
                    f"--size_x {dx} --size_y {dy} --size_z {dz} "
                    f"--flex {self.flex_receptor.filepath}"
                )
                commands.append(command)
                isomer_outputs.append(docked)
            mol_outputs.append(isomer_outputs)

        # Run all commands at once
        self.run_multi(
            commands,
            verbose=True,
            raise_on_failure=False,
            n_jobs=self.n_jobs.value,
        )

        # Convert each pose to SDF, update isomer conformation
        res, complexes = self._parse_flex_adv_outputs(
            mols, mol_outputs, receptor=self.receptor.filepath
        )
        self.out.send(mols)
        self.out_residues.send(list(res.values()))
        self.out_complex.send(list(complexes.values()))


def _clean_pdbqt_atomtypes(pdbqt_in: Path, pdbqt_out: Path) -> None:
    """Replaces ``G0`` and ``CG0`` atomtypes with normal carbons."""
    with pdbqt_in.open() as inp, pdbqt_out.open("w") as out:
        out.write(re.sub("(CG0)|(G0)", "C", inp.read()))


class VinaGPU(_Vina):
    """
    Runs Vina-GPU on a molecule input.

    See [#ding2023]_ and [#trott2010]_.

    The step expects to either find a ``vina`` executable in the ``PATH``,
    an appropriate module defined in ``config.toml``, or a module specified
    using the :attr:`~maize.core.node.Node.modules` attribute.

    Notes
    -----
    The interface is mostly the same as Vina's, but requires some additional handling
    of the custom compiled kernels, a small change in the commandline parameters, and
    allows for docking a directory of ligands at once. The source can be found
    `here <https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0>`_. Installation requires
    both the *boost* sources and installed headers, and ``-DOPENCL_3_0`` should *not*
    be specified (contrary to the official installation instructions).

    References
    ----------
    .. [#ding2023] Ding, J. et al. Vina-GPU 2.0: Further Accelerating AutoDock Vina
       and Its Derivatives with Graphics Processing Units. J. Chem. Inf. Model. (2023)
       doi:10.1021/acs.jcim.2c01504.

    .. [#trott2010] Trott, O. & Olson, A. J. AutoDock Vina: Improving the speed and accuracy
       of docking with a new scoring function, efficient optimization, and
       multithreading. Journal of Computational Chemistry 31, 455-461 (2010).

    """
    tags = {"chemistry", "docking", "scorer", "tagger"}

    required_callables = ["vinagpu"]
    """Requires the ``vinagpu`` executable"""

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko==0.4`` installed"""

    def run(self) -> None:
        mols = self.inp.receive()

        # VinaGPU requires custom built kernels in the same directory,
        # so we copy them from the install location
        kernel_dir = Path(self.runnable["vinagpu"]).parent
        self.logger.debug("Looking for kernels in '%s'", kernel_dir.as_posix())
        for i in (1, 2):
            kernel = kernel_dir / f"Kernel{i}_Opt.bin"
            if not kernel.exists():
                raise FileNotFoundError(
                    "VinaGPU requires the 'Kernel_Opt.bin' files to "
                    "be present in the Vina-GPU binary folder"
                )
            shutil.copy(kernel, self.work_dir)

        x, y, z = self.search_center.value
        dx, dy, dz = self.search_range.value

        lig_temp = Path("lig-temp.pdbqt")
        inputs, outputs = Path("inputs"), Path("outputs")
        inputs.mkdir()
        outputs.mkdir()
        mol_docked = []
        for i, mol in enumerate(mols):
            self.logger.info("Docking molecule %s: '%s'", i, mol)
            docked = []
            for j, isomer in enumerate(mol.molecules):
                self.logger.debug("  Docking isomer %s: '%s'", j, isomer)
                ligand = inputs / f"{isomer.inchi}.pdbqt"
                docked.append(outputs / f"{isomer.inchi}_out.pdbqt")
                try:
                    _mol2pdbqt(lig_temp, isomer)
                    _clean_pdbqt_atomtypes(lig_temp, ligand)
                except ValueError as err:
                    self.logger.warning(
                        "Skipping '%s' due to PDBQT conversion error:\n %s",
                        isomer.name or isomer.inchi,
                        err,
                    )
            mol_docked.append(docked)

        command = (
            f"{self.runnable['vinagpu']} --receptor {self.receptor.filepath.as_posix()} "
            f"--ligand_directory {inputs.as_posix()} "
            f"--output_directory {outputs.as_posix()} "
            f"--thread 8000 --seed {self.seed.value} "
            f"--num_modes {self.n_poses.value} "
            f"--center_x {x} --center_y {y} --center_z {z} "
            f"--size_x {dx} --size_y {dy} --size_z {dz} "
        )
        self.run_command(
            command,
            verbose=True,
            raise_on_failure=False,
            # validators=[SuccessValidator("...done")],
        )
        # Convert each pose to SDF, update isomer conformation
        self.logger.debug("Docking outputs: '%s'", list(outputs.iterdir()))
        self.logger.debug("Parsing: '%s'", mol_docked)
        self._parse_adv_outputs(mols, mol_docked)
        self.out.send(mols)


class QuickVinaGPU(_Vina):
    """
    Runs QuickVina2 or QuickVina-W for GPUs on a molecule input.

    For an overview, see `this <https://qvina.github.io/>`_, [#hassan2017]_ and [#alhossary2015]_.

    The step expects to either find a ``quickvina`` executable in the ``PATH``,
    an appropriate module defined in ``config.toml``, or a module specified
    using the :attr:`~maize.core.node.Node.modules` attribute.

    Notes
    -----
    The interface is mostly the same as Vina's, but requires some additional handling
    of the custom compiled kernels, a small change in the commandline parameters, and
    allows for docking a directory of ligands at once. The source can be found
    `here <https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0>`_. Installation requires
    both the *boost* sources and installed headers, and ``-DOPENCL_3_0`` should *not*
    be specified (contrary to the official installation instructions).

    References
    ----------
    .. [#hassan2017] Hassan, N. M., Alhossary, A. A., Mu, Y. & Kwoh, C.-K.
       Protein-Ligand Blind Docking Using QuickVina-W With Inter-Process
       Spatio-Temporal Integration. Sci Rep 7, 15451 (2017).

    .. [#alhossary2015] Alhossary, A., Handoko, S. D., Mu, Y. & Kwoh, C.-K.
       Fast, accurate, and reliable molecular docking with QuickVina 2.
       Bioinformatics 31, 2214-2216 (2015).

    """
    tags = {"chemistry", "docking", "scorer", "tagger"}

    required_callables = ["quickvina"]
    """Requires the ``quickvina`` executable"""

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko==0.4`` installed"""

    def run(self) -> None:
        mols = self.inp.receive()

        x, y, z = self.search_center.value
        dx, dy, dz = self.search_range.value

        # VinaGPU requires custom built kernels in the same directory,
        # so we copy them from the install location
        kernel_dir = Path(self.runnable["quickvina"]).parent
        self.logger.debug("Looking for kernels in '%s'", kernel_dir.as_posix())
        if not (kernel_dir / "Kernel2_Opt.bin").exists():
            raise FileNotFoundError(
                "VinaGPU requires the 'Kernel_Opt.bin' files to "
                "be present in the Vina-GPU binary folder"
            )
        shutil.copy(kernel_dir / "Kernel2_Opt.bin", self.work_dir / "Kernel2_Opt.bin")
        shutil.copytree(kernel_dir / "OpenCL", self.work_dir / "OpenCL")

        # Collect all docking commands to be executed, create directories
        lig_temp = Path("lig-temp.pdbqt")
        commands: list[str] = []
        mol_outputs: list[list[Path]] = []
        for i, mol in enumerate(mols):
            mol_path = Path(f"mol-{i}")
            mol_path.mkdir()
            isomer_outputs: list[Path] = []
            self.logger.info("Docking molecule %s: '%s'", i, mol)
            for j, isomer in enumerate(mol.molecules):
                self.logger.debug("  Docking isomer %s: '%s'", j, isomer)
                iso_path = mol_path / f"isomer-{j}"
                iso_path.mkdir()
                ligand = iso_path / "input.pdbqt"
                docked = iso_path / f"{isomer.inchi}_out.pdbqt"
                try:
                    _mol2pdbqt(lig_temp, isomer)
                    _clean_pdbqt_atomtypes(lig_temp, ligand)
                except ValueError as err:
                    self.logger.warning(
                        "Skipping '%s' due to PDBQT conversion error:\n %s",
                        isomer.name or isomer.inchi,
                        err,
                    )

                command = (
                    f"{self.runnable['quickvina']} --receptor {self.receptor.filepath.as_posix()} "
                    f"--ligand {ligand.as_posix()} "
                    f"--seed {self.seed.value} --out {docked.as_posix()} "
                    f"--thread 8000 "
                    f"--num_modes {self.n_poses.value} "
                    f"--center_x {x} --center_y {y} --center_z {z} "
                    f"--size_x {dx} --size_y {dy} --size_z {dz} "
                )
                commands.append(command)
                isomer_outputs.append(docked)
            mol_outputs.append(isomer_outputs)

        # Run all commands at once
        self.run_multi(
            commands,
            verbose=True,
            raise_on_failure=False,
            validators=[SuccessValidator("Writing output")],
            n_jobs=self.n_jobs.value,
        )

        # Convert each pose to SDF, update isomer conformation
        self._parse_adv_outputs(mols, mol_outputs)
        self.out.send(mols)


class VinaScore(Node):
    """
    Runs Vina scoring on a molecule input.

    The step expects to either find a ``vina`` executable in the ``PATH``,
    an appropriate module defined in ``config.toml``, or a module specified
    using the :attr:`~maize.core.node.Node.modules` attribute.

    """
    tags = {"chemistry", "docking", "scorer", "tagger"}

    SCORE_TAG = "energy"

    required_callables = ["vina"]
    """Requires the ``vina`` executable"""

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko==0.4`` installed"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to dock"""

    out: Output[list[IsomerCollection]] = Output(optional=True)
    """Molecules with scores attached."""

    out_scores: Output[NDArray[np.float32]] = Output()
    """Docking scores, the best for each docked IsomerCollection"""

    n_jobs: Parameter[int] = Parameter(default=cpu_count())
    """Number of docking runs to perform in parallel"""

    receptor: FileParameter[Annotated[Path, Suffix("pdbqt")]] = FileParameter()
    """Path to the receptor structure"""

    def run(self) -> None:
        mols = self.inp.receive()

        # Collect all docking commands to be executed, create directories
        lig_temp = Path("lig-temp.pdbqt")
        commands: list[str] = []
        ligands: list[str] = []
        for i, mol in enumerate(mols):
            self.logger.info("Scoring molecule %s: '%s'", i, mol)
            for j, isomer in enumerate(mol.molecules):
                self.logger.debug("  Scoring isomer %s: '%s'", j, isomer)
                ligand = Path(f"{isomer.inchi}_input.pdbqt")
                try:
                    _mol2pdbqt(lig_temp, isomer)
                    _clean_pdbqt_atomtypes(lig_temp, ligand)
                except ValueError as err:
                    self.logger.warning(
                        "Skipping '%s' due to PDBQT conversion error:\n %s", isomer, err
                    )

                command = (
                    f"{self.runnable['vina']} --receptor {self.receptor.filepath.as_posix()} "
                    f"--ligand {ligand.as_posix()} --autobox --score_only"
                )
                commands.append(command)
                ligands.append(isomer.inchi)

        # Run all commands at once
        results = self.run_multi(
            commands=commands,
            verbose=True,
            raise_on_failure=False,
            validators=[SuccessValidator("Estimated Free Energy")],
            n_jobs=self.n_jobs.value,
        )

        idx = 0
        for mol in mols:
            for isomer in mol.molecules:
                isomer.set_tag("score_type", "oracle")
                score = np.nan
                if isomer.inchi in ligands:
                    if match := re.search(SCORE_ONLY_RESULT_REGEX, results[idx].stdout.decode()):
                        score = float(match.group(1))
                    idx += 1
                isomer.add_score(self.SCORE_TAG, score)
                self.logger.info(
                    "Parsed isomer '%s', score %s",
                    isomer.name or isomer.inchi,
                    isomer.scores[self.SCORE_TAG],
                )
            mol.primary_score_tag = self.SCORE_TAG

        self.out_scores.send(np.array([mol.primary_score for mol in mols]))
        self.out.send(mols)


# 1UYD previously published with Icolos (IcolosData/molecules/1UYD)
@pytest.fixture
def protein_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_apo.pdb"


@pytest.fixture
def receptor_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_fixed.pdbqt"


# NNMT_G249N is public data
@pytest.fixture
def flex_receptor_path(shared_datadir: Path) -> Path:
    return shared_datadir / "NNMT_G249N_receptor_flex.pdbqt"


@pytest.fixture
def rigid_receptor_path(shared_datadir: Path) -> Path:
    return shared_datadir / "NNMT_G249N_receptor_rigid.pdbqt"


@pytest.fixture
def ligand_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_ligand.sdf"


# From AD GPU
@pytest.fixture
def grid_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1stp" / "1stp_protein.maps.fld"


@pytest.fixture
def grid_tar(shared_datadir: Path) -> Path:
    return shared_datadir / "1uyd.tar"


class TestSuiteAutodock:
    @pytest.mark.needs_node("preparepdbqt")
    def test_PreparePDBQT(
        self, temp_working_dir: Path, protein_path: Path, test_config: Config
    ) -> None:
        rig = TestRig(PreparePDBQT, config=test_config)
        params: list[dict[str, Any]] = [
            {"repairs": "None"},
            {
                "repairs": "bonds_hydrogens",
                "cleanup_protein": ["lps", "waters"],
                "remove_nonstd": True,
            },
            {"repairs": "checkhydrogens", "cleanup_protein": ["nphs", "nonstdres"]},
        ]
        for param in params:
            res = rig.setup_run(inputs={"inp": [protein_path]}, parameters=param)
            file = res["out"].get()
            assert file is not None
            assert file.exists()

    @pytest.mark.needs_node("autodockgpu")
    def test_AutoDockGPU(
        self, temp_working_dir: Path, grid_path: Path, ligand_path: Path, test_config: Config
    ) -> None:
        rig = TestRig(AutoDockGPU, config=test_config)
        mol = IsomerCollection.from_sdf(ligand_path)
        mol.embed()
        # SMILES from 1UYD data (Icolos)
        mol_fail = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol_fail.embed()
        res = rig.setup_run(
            parameters={"grid_file": grid_path, "derivtypes": {"NA": "N", "SA": "S"}},
            inputs={"inp": [[mol, mol_fail]]},
        )
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 2
        assert docked[0].scored
        assert not docked[1].scored
        assert docked[0].molecules[0].n_conformers == 20
        assert -5.0 < docked[0].primary_score < -2.0

    @pytest.mark.needs_node("autodockgpu")
    def test_AutoDockGPU_cons(
        self, temp_working_dir: Path, grid_tar: Path, ligand_path: Path, test_config: Config
    ) -> None:
        rig = TestRig(AutoDockGPU, config=test_config)
        mol = IsomerCollection.from_sdf(ligand_path)
        mol.embed()
        # SMILES from 1UYD data (Icolos)
        mol_fail = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol_fail.embed()
        res = rig.setup_run(
            parameters={"derivtypes": {"NA": "N", "SA": "S"}},
            inputs={"inp": [[mol, mol_fail]], "inp_grid": grid_tar},
        )
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 2
        assert docked[0].scored
        assert docked[1].scored
        assert docked[0].molecules[0].n_conformers == 20

    @pytest.mark.needs_node("vina")
    def test_Vina(self, temp_working_dir: Path, receptor_path: Path, test_config: Config) -> None:
        """Test Autodock in isolation"""
        rig = TestRig(Vina, config=test_config)
        params = {
            "search_center": (3.3, 11.5, 24.8),
            "receptor": receptor_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -11.0 < docked[0].primary_score < -7.0

    @pytest.mark.needs_node("vina")
    def test_VinaFlex(
        self,
        temp_working_dir: Path,
        flex_receptor_path: Path,
        rigid_receptor_path: Path,
        test_config: Config,
    ) -> None:
        """Test Autodock in isolation"""
        rig = TestRig(VinaFlex, config=test_config)
        params = {
            "search_center": (3.3, 11.5, 24.8),
            "receptor": rigid_receptor_path,
            "flex_receptor": flex_receptor_path,
            "n_poses": 4,
            "n_cores_per_job": 8,
        }
        mol = IsomerCollection.from_smiles("O=C(c1ccco1)N1CCCNCC1")
        mol.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers > 1
        prot = res["out_residues"].get()
        assert prot is not None
        assert len(prot) == 1
        assert prot[0].exists()

    @pytest.mark.needs_node("quickvinagpu")
    def test_QuickVinaGPU(
        self, temp_working_dir: Path, receptor_path: Path, test_config: Config
    ) -> None:
        """Test Autodock in isolation"""
        rig = TestRig(QuickVinaGPU, config=test_config)
        params = {
            "search_center": (3.3, 11.5, 24.8),
            "receptor": receptor_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -11.0 < docked[0].primary_score < -7.0

    @pytest.mark.needs_node("vinagpu")
    def test_VinaGPU(
        self, temp_working_dir: Path, receptor_path: Path, test_config: Config
    ) -> None:
        """Test Autodock in isolation"""
        rig = TestRig(VinaGPU, config=test_config)
        params = {
            "search_center": (3.3, 11.5, 24.8),
            "receptor": receptor_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -11.0 < docked[0].primary_score < -7.0

    @pytest.mark.needs_node("vinascore")
    def test_VinaScore(
        self, temp_working_dir: Path, receptor_path: Path, test_config: Config
    ) -> None:
        """Test Vina in isolation"""
        rig = TestRig(VinaScore, config=test_config)
        params = {"receptor": receptor_path}
        mol1 = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol1.embed()
        mol2 = IsomerCollection.from_smiles("Nc1nc(Cl)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol2.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol1, mol2]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 2
        assert all(dock.scored for dock in docked)
        scores = res["out_scores"].get()
        assert scores is not None
        assert len(scores) == 2
        assert (scores < 0).all()
