"""Autodock Vina implementation"""

# pylint: disable=import-outside-toplevel, import-error

import json
import logging
from pathlib import Path
import re
import shutil
import tarfile
from typing import Annotated, Any, Literal, Optional
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, FileParameter, Suffix, Input, Output
from maize.utilities.chem import Isomer, IsomerCollection
from maize.utilities.testing import TestRig
from maize.utilities.validation import SuccessValidator, FileValidator
from maize.utilities.resources import cpu_count

AD_HIGH_ENERGY = 1000
SCORE_ONLY_RESULT_REGEX = re.compile(r"\s*Estimated Free Energy of Binding\s*\:\s+(-?\d+\.\d+)\s*")


log = logging.getLogger("run")


def _adv_score_parser_meeko(props: dict[str, str]) -> float:
    """Parse scores from Vina output."""
    log.debug("Parsing SDF properties '%s'", props)
    value = float(json.loads(props.get("meeko", ""))["free_energy"])
    log.debug("Parsed value '%s' from properties", value)
    return value


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


# Helper functions using meeko to convert to PDBQT...
def _mol2pdbqt(file: Path, isomer: "Isomer") -> None:
    """Converts an isomer to a PDBQT file using meeko"""
    from meeko import MoleculePreparation

    preparator = MoleculePreparation()
    preparator.prepare(isomer._molecule)
    preparator.write_pdbqt_file(file)


# ...and from DLG (a kind of PDBQT file) to SDF
def _adv2sdf(inp_file: Path, sdf: Path) -> None:
    """Converts an AD DLG file to SDF using meeko"""
    from meeko import PDBQTMolecule, RDKitMolCreate

    with inp_file.open() as inp:
        mol = PDBQTMolecule(inp.read(), is_dlg=inp_file.suffix == ".dlg", skip_typing=True)
    with sdf.open("w") as sdfout:
        out, failures = RDKitMolCreate.write_sd_string(mol)
        sdfout.write(out)
    if len(failures) > 0:
        raise IOError(f"Meeko failed to write file '{sdf.as_posix()}'")


class PreparePDBQT(Node):
    """Prepares a receptor for docking with Vina."""

    required_callables = ["prepare_receptor"]
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
        self.out.send(receptor_pdbqt)


# TODO Allow anchored / constrained docking, see:
# https://github.com/ccsb-scripps/AutoDock-GPU/wiki/Anchored-docking
class PrepareGrid(Node):
    """Prepares a receptor for docking with AutoDock4."""

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

    def run(self) -> None:
        structure = self.inp_structure.receive()

        # Create receptor PDBQT (if needed)
        receptor_pdbqt = Path("rec.pdbqt")
        if structure.suffix == ".pdbqt":
            shutil.move(structure, receptor_pdbqt)
        else:
            self.run_command(
                f"{self.runnable['prepare_receptor']} "
                f"-r {structure.as_posix()} "
                f"-o {receptor_pdbqt.as_posix()}",
                validators=[FileValidator(receptor_pdbqt)],
            )

        # Create temporary ligand PDBQT
        if self.inp_ligand.ready():
            lig = self.inp_ligand.receive()
            lig_pdbqt = Path("lig.pdbqt")
            _mol2pdbqt(lig_pdbqt, lig)
            command = (
                f"{self.runnable['write_gpf']} "
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
        glg = gpf.with_suffix(".glg")
        fld = glg.with_suffix(".maps.fld")
        self.run_command(
            f"{self.runnable['autogrid']} -p {gpf.as_posix()} -l {glg.as_posix()}",
            validators=[FileValidator(glg), FileValidator(fld)],
        )

        # Wrap it all up
        tar = Path("grid.tar")
        with tarfile.open(tar, "w") as archive:
            for file in Path().glob("*.map"):
                archive.add(file)
            for file in (receptor_pdbqt, gpf, glg, fld):
                archive.add(file)
        self.out.send(tar)


class AutoDockGPU(Node):
    """
    Runs AutoDock on the GPU [#santos2021]_.

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

    required_callables = ["autodock_gpu"]
    """Requires the ``autodock_gpu`` executable"""

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko==0.4`` installed"""

    inp: Input[list[IsomerCollection]] = Input()
    """
    List of molecules to dock, each molecule can have multiple isomers,
    these will be docked separately.

    """

    out: Output[list[IsomerCollection]] = Output(optional=True)
    """
    Docked molecules with conformations and scores attached. Also
    include per-conformer clustering information performed by
    AutoDock, use the keys 'rmsd', 'cluster_rmsd', 'cluster' to access.

    """

    out_scores: Output[NDArray[np.float32]] = Output()
    """Docking scores, the best for each docked IsomerCollection"""

    ref_ligand: Parameter[Isomer] = Parameter(optional=True)
    """Optional reference ligand for RMSD analysis"""

    grid_file: FileParameter[Path] = FileParameter()
    """The protein grid file, all internally referenced files must be available"""

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

    def run(self) -> None:
        mols = self.inp.receive()

        # Molecule inputs
        inputs = Path("inputs")
        inputs.mkdir()

        # Convert all ligands to pdbqt and collect
        # their paths and names in a batch file
        batch_file = Path("batch.txt")
        with batch_file.open("w") as file:
            file.write(f"{self.grid_file.filepath.as_posix()}\n")
            for mol in mols:
                for isomer in mol.molecules:
                    # Tools like REINVENT rely on getting the same number of scores out
                    # as molecules, so we can't filter out failed embeddings earlier...
                    if isomer.n_conformers == 0:
                        self.logger.warning(
                            "No embedding for '%s' ('%s'), skipping...",
                            isomer.inchi,
                            isomer.to_smiles(),
                        )
                        continue

                    ligand = inputs / f"{isomer.inchi}.pdbqt"

                    # Create pdbqt input
                    _mol2pdbqt(ligand, isomer)

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

        # Possible derivtypes
        if self.derivtypes.value:
            derivtypes = "/".join(f"{key}={value}" for key, value in self.derivtypes.value.items())
            command += f" --derivtype {derivtypes}"

        validators = [SuccessValidator("All jobs ran without errors")] if self.strict.value else []
        with self.gpus(1):
            self.run_command(command, verbose=True, validators=validators)

        # Collect outputs
        for mol in mols:
            for isomer in mol.molecules:
                isomer.score_tag = "energy"
                output = Path(f"{isomer.inchi}.xml")

                # Failed dockings and missing embeddings get a NaN
                try:
                    # Isomer scores are in the order of the indices
                    results = _adgpu_score_parser(output, log=self.logger)
                except (KeyError, FileNotFoundError) as err:
                    if self.strict.value:
                        raise err
                    self.logger.warning("Docking isomer '%s' failed", isomer.inchi)
                    isomer.scores = np.full(self.nrun.value, np.nan)
                    continue

                res_transpose = _list_of_dicts2dict_of_lists(list(results.values()))

                # High energies indicate grid problems
                if any(ener > AD_HIGH_ENERGY for ener in res_transpose["energy"]):
                    self.logger.warning(
                        "Isomer '%s' ('%s') has runs with high energy poses. This indicates "
                        "a possible lack of grid coverage or a poorly-defined search space. "
                        "Adjust `search_range` and `search_center` during grid preparation.",
                        isomer.inchi,
                        mol.smiles,
                    )

                # Convenience score attribute
                isomer.set_tag("score_type", "oracle")
                isomer.scores = np.array(res_transpose["energy"])
                self.logger.info("Parsed isomer '%s', score %s", isomer.inchi, isomer.scores.min())

                # We only parse the conformers if the user asks for them,
                # otherwise it unnecessarily slows things like REINVENT down
                if not self.scores_only.value:
                    # This allows us to convert all pdbqt outputs
                    # into one SDF, with the scoring order
                    sdf_out = Path(f"{isomer.inchi}-out.sdf")
                    _adv2sdf(output.with_suffix(".dlg"), sdf_out)

                    # Add all conformers and set their coords
                    isomer.update_conformers_from_sdf(sdf_out)

                    # AD gives us lots of useful information for each conformer, e.g. energy
                    # reference RMSD and cluster, we tag each conformer with this information
                    for key, vals in res_transpose.items():
                        for conf, val in zip(isomer.conformers, vals):
                            conf.set_tag(key, val)

        if not self.scores_only.value:
            self.out.send(mols)
        self.out_scores.send(np.array([mol.best_score for mol in mols]))


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

    def prepare(self) -> None:
        super().prepare()
        import meeko

        if meeko.__version__ == "0.5.0":
            raise ImportError(
                "Vina nodes are incompatible with meeko 0.5.0 due "
                "to an upstream parsing issue with PDBQT files"
            )

    def _parse_adv_outputs(
        self, mols: list[IsomerCollection], mol_outputs: list[list[Path]]
    ) -> None:
        """Parses ADV output, including conformers and scores from PDBQT or DLG outputs"""
        moldict = {iso.inchi: iso for mol in mols for iso in mol.molecules}
        outdict = {file.stem.strip("_out"): file for folder in mol_outputs for file in folder}
        for i, (key, file) in enumerate(outdict.items()):
            isomer = moldict[key]
            self.logger.info("Parsing isomer %s: '%s'", i, isomer)
            if not file.exists() or file.stat().st_size == 0:
                self.logger.warning("Docking failed for '%s' (%s)", isomer.inchi, isomer)
                continue
            _adv2sdf(file, file.with_suffix(".sdf"))
            isomer.update_conformers_from_sdf(
                file.with_suffix(".sdf"), score_parser=_adv_score_parser_meeko
            )


class Vina(_Vina):
    """
    Runs Vina [#eberhardt2021]_ on a molecule input.

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

    required_callables = ["vina"]
    """Requires the ``vina`` executable"""

    required_packages = ["meeko"]
    """Requires a custom environment with ``meeko==0.4`` installed"""

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
                        "Skipping '%s' due to PDBQT conversion error:\n %s", isomer, err
                    )

                command = (
                    f"{self.runnable['vina']} --receptor {self.receptor.filepath.as_posix()} "
                    f"--ligand {ligand.as_posix()} "
                    f"--cpu 1 --seed {self.seed.value} --out {docked.as_posix()} "
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
            n_jobs=self.n_jobs.value,
        )

        # Convert each pose to SDF, update isomer conformation
        self._parse_adv_outputs(mols, mol_outputs)
        self.out.send(mols)


def _clean_pdbqt_atomtypes(pdbqt_in: Path, pdbqt_out: Path) -> None:
    """Replaces ``G0`` and ``CG0`` atomtypes with normal carbons."""
    with pdbqt_in.open() as inp, pdbqt_out.open("w") as out:
        out.write(re.sub("(CG0)|(G0)", "C", inp.read()))


class VinaGPU(_Vina):
    """
    Runs Vina-GPU [#ding2023]_ on a molecule input.

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
                        "Skipping '%s' due to PDBQT conversion error:\n %s", isomer, err
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
    Runs QuickVina2 or QuickVina-W for GPUs [#ding2023]_ on a molecule input.
    For an overview, see `this <https://qvina.github.io/>`_.

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
                        "Skipping '%s' due to PDBQT conversion error:\n %s", isomer, err
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
    Runs Vina scoring [#eberhardt2021]_ on a molecule input.

    The step expects to either find a ``vina`` executable in the ``PATH``,
    an appropriate module defined in ``config.toml``, or a module specified
    using the :attr:`~maize.core.node.Node.modules` attribute.

    """

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
                isomer.score_tag = "energy"
                isomer.set_tag("score_type", "oracle")
                score = np.nan
                if isomer.inchi in ligands:
                    if match := re.search(SCORE_ONLY_RESULT_REGEX, results[idx].stdout.decode()):
                        score = float(match.group(1))
                    idx += 1
                isomer.set_tag("energy", score)
                self.logger.info("Parsed isomer '%s', score %s", isomer.inchi, isomer.scores.min())

        self.out_scores.send(np.array([mol.best_score for mol in mols]))
        self.out.send(mols)


# 1UYD previously published with Icolos (IcolosData/molecules/1UYD)
@pytest.fixture
def protein_path(shared_datadir: Any) -> Any:
    return shared_datadir / "1UYD_apo.pdb"


@pytest.fixture
def receptor_path(shared_datadir: Any) -> Any:
    return shared_datadir / "1UYD_fixed.pdbqt"


@pytest.fixture
def ligand_path(shared_datadir: Any) -> Any:
    return shared_datadir / "1UYD_ligand.sdf"


# From AD GPU
@pytest.fixture
def grid_path(shared_datadir: Any) -> Any:
    return shared_datadir / "1stp" / "1stp_protein.maps.fld"


class TestSuiteAutodock:
    def test_PreparePDBQT(self, temp_working_dir: Any, protein_path: Any, test_config: Any) -> None:
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

    def test_AutoDockGPU(
        self, temp_working_dir: Any, grid_path: Any, ligand_path: Any, test_config: Any
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
        assert -5.0 < docked[0].best_score < -2.0

    def test_Vina(self, temp_working_dir: Any, receptor_path: Any, test_config: Any) -> None:
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
        assert -11.0 < docked[0].best_score < -7.0

    def test_QuickVinaGPU(
        self, temp_working_dir: Any, receptor_path: Any, test_config: Any
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
        assert -11.0 < docked[0].best_score < -7.0

    def test_VinaGPU(self, temp_working_dir: Any, receptor_path: Any, test_config: Any) -> None:
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
        assert -11.0 < docked[0].best_score < -7.0

    def test_VinaScore(self, temp_working_dir: Any, receptor_path: Any, test_config: Any) -> None:
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
