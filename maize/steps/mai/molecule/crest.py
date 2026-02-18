from pathlib import Path
from typing import Any, cast, Literal
from subprocess import CompletedProcess
from rdkit import Chem
from rdkit.Chem import rdMolAlign, TorsionFingerprints, AllChem
from rdkit.ML.Cluster import Butina as butina_clust
import json
import copy
import shutil
import os
import time
import numpy as np
import logging
import pytest

from maize.utilities.testing import TestRig

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Flag, FileParameter

from maize.utilities.chem import Isomer, IsomerCollection, Conformer
from maize.steps.mai.molecule.compchem_utils import Structure, EntryCoord, Loader


log = logging.getLogger("run")


def create_constraints_crest(isomer: Isomer, path: Path) -> str:
    """
    Create constraint file for CREST calculations starting from the isomer
    object and the constrained tag associated. Need to specify the constrained
    and unconstrained files, as well as a reference geometry. For reference see:
    https://crest-lab.github.io/crest-docs/page/examples/example_4.html

    """
    constr_indexes = json.loads(isomer.tags["constraints"])
    isomer_name = isomer.get_tag("logging_name")
    suffix = "xyz" if isomer.has_tag("connectivity") else "sdf"

    constraint_file = "{}/{}_crest_constraints.inp".format(path, isomer_name)
    unconstrained_indexes = [
        i for i in list(range(1, isomer.n_atoms + 1)) if i not in constr_indexes
    ]

    ## return list of uncostrained indexes in the format accepted by CREST
    ranges = []
    start = end = unconstrained_indexes[0]

    for num in unconstrained_indexes[1:]:
        if num == end + 1:
            end = num
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = num

    # Append the last range after the loop ends
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")

    uncostrained_string = ", ".join(ranges)

    with open(constraint_file, "w") as f:
        f.write("{}\n".format("$constrain"))
        f.write("{}\n".format("   atoms: " + str(constr_indexes).replace("[", "").replace("]", "")))
        f.write("{}\n".format("   force constant=0.5 "))
        f.write("{}\n".format(f"   reference={isomer_name}_crest_inp.{suffix}"))
        f.write("{}\n".format("$metadyn:"))
        f.write("{}\n".format("   atoms: " + uncostrained_string))
        f.write("{}\n".format("$end"))

    return constraint_file


def update_conformers_from_xyz(iso: Isomer, xyz: Path | None = None) -> Isomer:
    """
    Update molecule conformers from an Crest XYZ output file.

    Parameters
    ----------
    iso
        isomer object
    xyz
        The xyz file to initialize the molecule with

    Returns
    -------
    Isomer
        updated isomer objects

    Raises
    ------
    ChemistryException
        If there was an error parsing the xyz

    """

    atom_string: list[str] = [
        "H",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Si",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "Ni",
        "Ir",
    ]

    iso.clear_conformers()
    if xyz:
        with open(xyz, "r") as f:
            crest_energies_dict = {}
            read_coords = False
            read_energy = False
            conformer_count = 0
            conformer_energy = np.nan
            number_atoms = 0
            xyz_str_single = ""
            atom_id = 1
            for line in f.readlines():
                line_lst = line.split()
                if read_coords and line_lst[0] in atom_string:
                    xyz_str_single += line
                    atom_id += 1
                elif read_energy:
                    conformer_energy = float(line_lst[0])
                    xyz_str_single += line
                    read_coords = True
                    read_energy = False
                elif len(line_lst) == 1 and not read_energy:
                    if number_atoms != 0:
                        try:
                            conf = Conformer.from_rdmol(
                                Chem.rdmolfiles.MolFromXYZBlock(xyz_str_single),
                                parent=iso,
                                renumber=False,
                                sanitize=False,
                            )
                            iso.add_conformer(conf)
                            crest_energies_dict[conformer_count] = conformer_energy
                            conformer_count += 1
                        except ValueError as err:
                            log.info("Unable to parse conformer, error: %s", err)
                            continue
                    number_atoms = int(line_lst[0])

                    atom_id = 1
                    read_coords = False
                    read_energy = True
                    xyz_str_single = line
            if number_atoms != 0:
                try:
                    conf = Conformer.from_rdmol(
                        Chem.rdmolfiles.MolFromXYZBlock(xyz_str_single),
                        parent=iso,
                        renumber=False,
                        sanitize=False,
                    )
                    iso.add_conformer(conf)
                    crest_energies_dict[conformer_count] = conformer_energy
                    conformer_count += 1
                except ValueError as err:
                    log.warning("Unable to parse conformer, error: %s", err)
        iso.set_tag("crest_energies", json.dumps(crest_energies_dict))
        log.info(f"found {str(conformer_count)} conformers for {xyz}")
        return iso
    else:
        raise FileNotFoundError("Could not find the CREST conformers file")


class Crest(Node):
    """
    Runs Crest conformational sampling semiempirical method on Isomer class.

    References
    ----------
    API documentation: https://xtb-docs.readthedocs.io/en/latest/contents.html
    Key citation reference for the XTB methods and current implementation:
    {C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
    S. Spicher, S. Grimme WIREs Comput. Mol. Sci., 2020, 11, e01493. DOI: 10.1002/wcms.1493}

    """

    required_callables = ["crest"]

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    method: Parameter[str] = Parameter(default="gfn2")
    """Accuracy/speed of the simulation"""

    fold: FileParameter[Path] = FileParameter(optional=True)
    """path of folder to dump results outputs"""

    batch: Flag = Flag(default=True)
    """Flag to submit to SLURM queueing system"""

    n_jobs: Parameter[int] = Parameter(default=100)
    """Number of parallel processes to use"""

    n_threads_job: Parameter[int] = Parameter(default=2)
    """Number of parallel processes to use"""

    e_window: Parameter[float] = Parameter(default=3.0)
    """Energy (kcal/mol) threshold for conformer selection. See -ewin flag in Crest Documentation"""

    charge: Parameter[int] = Parameter(default=0)
    """Charge of the molecule"""

    multiplicity: Parameter[int] = Parameter(default=1)
    """Multiplicity of the molecule"""

    label: Parameter[str] = Parameter(default="Crest_calculation")
    """Name for the job to be used in the logging."""

    solvent: Parameter[str] = Parameter(default="ether")
    """Solvent for the alpb implicit solvation model"""

    def _parse_crest_outputs(
        self,
        mols: list[IsomerCollection],
        mol_outputs: list[list[Path]],
        results: list[CompletedProcess[bytes]],
    ) -> None:
        """
        Parses crest output

        Parameters
        ----------
        mols
            List of IsomerCollection objects corresponding to the molecules in
            the calculation
        mol_outputs
            list containing list of paths for individual calculation output files
        results
            Results of the jobs
        """

        moldict = {iso.get_tag("crest_idx"): iso for mol in mols for iso in mol.molecules}
        count = 0

        if not os.path.exists(self.fold.value):
            os.mkdir(self.fold.value)

        for i, mol_folder in enumerate(mol_outputs):
            mol_path = self.fold.value / f"mol-{i}"
            if os.path.exists(mol_path):
                shutil.rmtree(mol_path)
            os.mkdir(mol_path)

            for j, iso_dirname in enumerate(mol_folder):
                isomer = moldict[f"{i}_{j}"]
                isomer.set_tag("crest_exit_code", 1)
                self.logger.info(isomer.get_tag("logging_name"))

                iso_output = iso_dirname / "crest_conformers.xyz"
                iso_stdout = results[count].stdout.decode()

                with open(
                    iso_dirname / f"{isomer.get_tag('logging_name')}_crest_out.txt", "w"
                ) as out:
                    out.write(iso_stdout)

                if not iso_output.exists():
                    self.logger.warning("Crest failed for '%s'", isomer.get_tag("logging_name"))
                    continue

                isomer = update_conformers_from_xyz(isomer, iso_output)

                # loop through the standard output and search for normal crest termination.
                pattern = "CREST terminated normally."
                for line in iso_stdout.split("\n"):
                    if pattern in line:
                        isomer.set_tag("crest_exit_code", 0)

                try:
                    energy_best = isomer.get_tag("crest_energies")
                except KeyError:
                    energy_best = f"error in loading the energy for {iso_output}"
                self.logger.info(energy_best)
                shutil.copytree(iso_dirname, mol_path / f"{isomer.get_tag('logging_name')}")

                count += 1

    def run(self) -> None:
        mols = self.inp.receive()
        commands: list[str] = []
        iso_paths: list[Path] = []
        mol_outputs: list[list[Path]] = []

        for i, mol in enumerate(mols):

            try:
                logging_name = mol.molecules[0].get_tag("component")
            except:
                logging_name = self.label.value

            mol_path = Path(f"mol-{i}").absolute()
            self.logger.debug(f" this is the path received by CREST {mol_path}")
            mol_path.mkdir()
            isomer_outputs: list[Path] = []
            self.logger.info(f"CREST calculations for molecule {i}: {logging_name}")

            for j, isomer in enumerate(mol.molecules):

                self.logger.info(f"{logging_name}: CREST similuation for isomer {j}: {isomer}")
                isomer.set_tag("crest_idx", f"{i}_{j}")
                isomer.set_tag("logging_name", logging_name)

                if (
                    isomer.has_tag("parameters")
                    and len(cast(list[Any], isomer.get_tag("parameters"))) > 1
                ):
                    isomer_charge = int(cast(list[Any], isomer.get_tag("parameters"))[0])
                    isomer_mult = int(cast(list[Any], isomer.get_tag("parameters"))[1])
                else:
                    isomer_charge = self.charge.value
                    isomer_mult = self.multiplicity.value

                iso_path = mol_path / f"isomer-{j}"
                iso_path.mkdir()
                iso_paths.append(iso_path)

                # check for constraints to apply to the calculations
                if isomer.has_tag("constraints"):
                    constraints = (
                        "--cinp " + create_constraints_crest(isomer, iso_path) + " --subrmsd"
                    )
                    self.logger.info(f"found constraint {constraints} for isomer {j}")
                else:
                    constraints = ""
                    # self.logger.info(f"no constraint for isomer {j}")
                output_dirname = iso_path

                # writes sdf input file for the crest calculation
                if isomer.has_tag("connectivity"):
                    if isomer.has_tag("g16_mm_geometries"):
                        self.logger.info(
                            "loading serialised geom from UFF optimisation on gaussian."
                        )
                        json_string = json.loads(cast(str, isomer.get_tag("g16_mm_geometries")))
                        mm_json = json_string[str(j)]
                        mm_iso = Structure(
                            [
                                EntryCoord(element=ec["element"], coords=ec["coords"]).to_Atom()
                                for ec in mm_json
                            ]
                        )
                        input_flname = f"{isomer.get_tag('logging_name')}_crest_inp.xyz"
                        input_path = iso_path / input_flname
                        mm_iso.write_xyz(input_path)

                    elif isomer.has_tag("gfnff_geometries"):
                        self.logger.info("loading serialised geom from GFN-FF optimisation on xtb.")
                        json_string = json.loads(cast(str, isomer.get_tag("gfnff_geometries")))
                        gfnff_json = json_string[str(j)]
                        gfnff_iso = Loader.molecule_from_json(gfnff_json, f"isomer-{j}")
                        input_flname = f"{isomer.get_tag('logging_name')}_crest_inp.xyz"
                        input_path = iso_path / input_flname
                        gfnff_iso.write_xyz(input_path)

                    else:
                        input_flname = f"{isomer.get_tag('logging_name')}_crest_inp.sdf"
                        input_path = iso_path / input_flname
                        isomer.to_sdf(path=input_path)

                else:
                    input_flname = f"{isomer.get_tag('logging_name')}_crest_inp.sdf"
                    input_path = iso_path / input_flname
                    isomer.to_sdf(path=input_path)

                # writes keywords for specific commands for the
                # calculation related to the compound properties
                keywords = (
                    f"--{self.method.value} -T {self.n_threads_job.value} "
                    f"-c {str(int(isomer_charge))} -u {str(int(isomer_mult - 1))} --alpb {str(self.solvent.value)} --squick --ewin {str(self.e_window.value)} --ethr 0.15 --noopt"
                )
                if isomer.has_tag("constraints"):
                    keywords += " --noreftopo --nci"
                command = f"{self.runnable['crest']} {input_path} {keywords}"
                commands.append(command)
                self.logger.info(command)

                isomer_outputs.append(output_dirname)

            mol_outputs.append(isomer_outputs)

        # Run all commands at once
        results = self.run_multi(
            commands,
            working_dirs=iso_paths,
            verbose=False,
            raise_on_failure=False,
            n_jobs=self.n_jobs.value,
        )

        # Convert each pose to SDF, update isomer conformation
        self._parse_crest_outputs(mols, mol_outputs, results)
        self.out.send(mols)


class ButinaClust(Node):
    """Butina clustering for conformations. Extracts the most representative conformers."""

    tags = {"chemistry", "clustering", "conformers"}

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    method: Parameter[Literal["rmsd", "tfd"]] = Parameter(default="rmsd")
    """Metric used for clustering conformers. RMSD or TFD."""

    label: Parameter[str] = Parameter(default="Butina_clustering")
    """Name for the job to be used in the logging."""

    n_conf: Parameter[int] = Parameter(default=5)
    """Number of centroids to be selected."""

    def run(self) -> None:

        isomercollections = self.inp.receive()

        for isomercollection in isomercollections:
            for i, isomer in enumerate(isomercollection.molecules):
                self.logger.info(
                    f"Isomer {i} conformers before filtering: {len(isomer.conformers)}"
                )
                rdmol = isomer._molecule
                mol_copy = copy.deepcopy(rdmol)
                mol = Chem.RemoveAllHs(mol_copy)

                if isomer.has_tag("XTB_geometries"):
                    self.logger.info("Filtering XTB re-optimized conformers")
                    mol.RemoveAllConformers()
                    conf_geometries_tag = json.loads(cast(str, isomer.get_tag("XTB_geometries")))
                    self.logger.info(conf_geometries_tag)
                    conf_ids = list(range(len(conf_geometries_tag)))
                    self.logger.info(f"list len of conformers tags: {conf_ids}")

                    for index in conf_ids:
                        self.logger.info(f"index: {index}")
                        conf_tag = conf_geometries_tag[str(index)]
                        if isinstance(conf_tag, str):
                            self.logger.info(
                                f"Conformer calculation failed for: conformer {index}. Skipping to the next one."
                            )
                            continue

                        else:
                            conf_structure = Loader.molecule_from_json(
                                conf_tag, f"conformer-{index}"
                            )
                            conf_mol = conf_structure.structure_to_rdmol(removeHs=True)
                            mol.AddConformer(conf_mol.GetConformer(), assignId=True)

                else:
                    self.logger.info("Filtering CREST generated conformers")
                    conf_ids = list(range(mol.GetNumConformers()))
                self.logger.info("starting ...")
                maxd = -100
                for j in range(0, 5):

                    for i in range(j, len(mol.GetConformers())):

                        d1 = rdMolAlign.AlignMol(mol, mol, prbCid=conf_ids[i], refCid=conf_ids[j])
                        start = time.time()

                        d2 = rdMolAlign.GetBestRMS(mol, mol, prbId=conf_ids[i], refId=conf_ids[j])
                        finish = time.time()
                        self.logger.info(f"time taken: {finish-start}s")
                        delt = d1 - d2
                        if delt < -1e-5:
                            self.logger.info(f"ooops, {i}, {delt}")
                        if delt > maxd:
                            maxd = delt
                            maxi = i
                            maxj = j

                d1 = rdMolAlign.AlignMol(mol, mol, prbCid=conf_ids[maxi], refCid=conf_ids[maxj])
                d2 = rdMolAlign.GetBestRMS(mol, mol, prbId=conf_ids[maxi], refId=conf_ids[maxj])

                self.logger.info("doing matrixes ...")
                if self.method.value == "tfd":
                    dmat = TorsionFingerprints.GetTFDMatrix(mol)
                    threshold = 0.05
                else:
                    dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
                    threshold = 0.75

                self.logger.info("doing clusters ...")
                clusts = butina_clust.ClusterData(
                    dmat, len(mol.GetConformers()), threshold, isDistData=True, reordering=True
                )
                centroids = [x[0] for x in clusts[: self.n_conf.value]]
                self.logger.info(
                    f"Selected conformers number {', '.join(str(p) for p in centroids) }, for isomer {isomer.name}"
                )

                if isomer.has_tag("XTB_geometries"):

                    centroid_selection = {}
                    for i, centroids_id in enumerate(centroids):
                        centroid_tag = conf_geometries_tag[str(centroids_id)]
                        centroid_selection[i] = centroid_tag

                    self.logger.info(
                        f"Isomer conformers after filtering: {len(centroid_selection)}"
                    )
                    isomer.set_tag("Filtered_conformers", json.dumps(centroid_selection))

                else:
                    original_confs = len(isomer.conformers)
                    self.logger.info(f"there are {original_confs} conformers before filtering")

                    saved_confs = []
                    for idx in list(range(original_confs)):
                        if idx in centroids:
                            self.logger.info(f"saving conformer {idx}")
                            saved_confs.append(isomer.conformers[idx])

                    isomer.clear_conformers()
                    for saved in saved_confs:
                        isomer.add_conformer(saved)

                    self.logger.info(f"Isomer conformers after filtering: {len(isomer.conformers)}")

        self.out.send(isomercollections)


class TestSuiteCrest:
    @pytest.mark.needs_node("crest")
    def test_Crest(
        self,
        temp_working_dir: Any,
        test_config: Any,
    ) -> None:
        rig = TestRig(Crest, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in ["OC(C)C", "CN(C)CO"]]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs]})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 2
        for mol in mols:
            assert len((mol.molecules[0].conformers)) >= 1
            assert mol.molecules[0].tags["crest_idx"]
            assert mol.molecules[0].tags["crest_energies"]
