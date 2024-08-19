from pathlib import Path
from typing import Any, List, Dict, cast
from typing_extensions import TypedDict
from subprocess import CompletedProcess
import re
import json
import logging
import shutil
import os
import pytest
import numpy as np


from maize.utilities.testing import TestRig
from maize.steps.mai.molecule.compchem_utils import Loader, atom_string
from maize.core.node import Node
from maize.steps.io import LoadData, Return
from maize.steps.mai.molecule import Smiles2Molecules
from maize.steps.mai.molecule.crest import Crest
from maize.core.workflow import Workflow
from maize.core.interface import Input, Output, Parameter, Flag, FileParameter
from maize.utilities.chem import Isomer, IsomerCollection
from maize.utilities.execution import JobResourceConfig


log = logging.getLogger("run")

SP_ENERGY_REGEX = re.compile(r"\s*TOTAL ENERGY\s+(-?\d+\.\d+)\s*")
SP_GRADIENT_REGEX = re.compile(r"\s*GRADIENT NORM\s+(-?\d+\.\d+)\s*")

AtomType = dict[str, str | int | list[float]]
AtomEntry = TypedDict("AtomEntry", {"element": str, "atom_id": int, "coords": list[float]})
ConfTag = TypedDict("ConfTag", {"atoms": list[AtomEntry], "energy": float, "gradient": float})


def create_constraints_xtb(iso: Isomer, path: Path) -> str:
    """
    Create constraint file for XTB calculations starting from
    the isomer object and the constrained tag associated

    """
    isomer_name = iso.get_tag('component')
    constr_indexes = iso.tags["constraints"].replace("[", "").replace("]", "")
    constraint_file = "{}/{}_xtb_constraints.inp".format(os.path.abspath(path), isomer_name)

    with open(constraint_file, "w") as f:
        f.write("{}\n".format("$fix"))
        f.write("{}\n".format("   atoms: " + constr_indexes))
        f.write("{}\n".format("$end"))

    return constraint_file


def _xtb_energy_parser_sp(stdout: str) -> float:
    """
    Parse energy from xtb output.

    Parameters
    ----------
    stdout
        string with path of stdout

    Returns
    -------
    float
        energy value
    """
    res = re.search(SP_ENERGY_REGEX, stdout)
    if res:
        return float(res.group(1))
    return np.nan


def _xtb_gradient_parser_sp(stdout: str) -> float:
    """
    Parse Gradient from xtb output.

    Parameters
    ----------
    stdout
        string with path of stdout

    Returns
    -------
    float
        gradient value
    """
    res = re.search(SP_GRADIENT_REGEX, stdout)
    if res:
        return float(res.group(1))
    return np.nan


class Xtb(Node):
    """
    Runs XTB semiempirical method on IsomerCollection class.

    Currently, the capabilities are
        * Performing geometry optimisation at GFN2-xTB level using approximate normal
          coordinate rational function optimizer (ANCopt)
        * Calculate partial atomic charges (Mulliken)
        * Calculate Wieberg partial bond orders
        * Return optimised 3D coordinates with the final energy and optimisation trajectory points

    References
    ----------
    API documentation: https://xtb-docs.readthedocs.io/en/latest/contents.html
    Key citation reference for the XTB methods and current implementation:
    {C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
    S. Spicher, S. Grimme WIREs Comput. Mol. Sci., 2020, 11, e01493. DOI: 10.1002/wcms.1493}

    """

    required_callables = ["xtb"]

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    opt: Parameter[bool] = Parameter(default=True)
    """Enable geometry optimisation"""

    charge: Parameter[int] = Parameter(default=0)
    """Charge of the molecule"""

    multiplicity: Parameter[int] = Parameter(default=1)
    """Multiplicity of the molecule"""

    solvent: Parameter[str] = Parameter(default="acetonitrile")
    """Solvent for the gbsa implicit solvation model"""

    fold: FileParameter[Path] = FileParameter(optional=True)
    """path of folder to dump results outputs"""

    batch: Flag = Flag(default=True)
    """Flag to submit to SLURM queueing system"""

    n_jobs: Parameter[int] = Parameter(default=100)
    """Number of parallel processes to use"""

    n_threads_job: Parameter[int] = Parameter(default=2)
    """Number of parallel processes to use"""


    _ENERGY_REGEX = re.compile(r"energy:\s*(-?\d+\.\d+)")
    _GRADIENT_REGEX = re.compile(r"gnorm:\s*(-?\d+\.\d+)")

    @staticmethod
    def get_atomic_charges(iso_dirname: Path) -> List[Dict[str, float]]:
        """
        gets atomic charge of individual atoms in molecule

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files

        Returns
        -------
        List[Dict[str, float]]
            list of dictionaries contaning label of atomic element and
            respective atomic charge
        """
        charges_file = iso_dirname / "charges"
        atom_id = 1
        charges = []
        with open(charges_file, "r") as f:
            for line in f.readlines():
                if len(line) != 0:
                    charges.append({"atom_id": atom_id, "charge": float(line)})
                    atom_id += 1
        if not charges:
            raise ValueError("Charges not found in molecule")
        return charges

    @staticmethod
    def get_wieberg_bo(iso_dirname: Path) -> List[Dict[str, float]]:
        """
        Gets wieberg bond orders for Bonds in molecule

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files

        Returns
        -------
        List[Dict[str, float]]
            list of dictionaries contaning label of atomic elements and
            respective bond order
        """
        charges_file = iso_dirname / "wbo"
        wbos = []
        with open(charges_file, "r") as f:
            for line in f.readlines():
                line_lst = line.split()
                if len(line_lst) == 3:
                    wbos.append(
                        {
                            "atom_id1": int(line_lst[0]),
                            "atom_id2": int(line_lst[1]),
                            "wbo": float(line_lst[2]),
                        }
                    )
        if not wbos:
            raise ValueError("WBOs not found in molecule")
        return wbos

    @staticmethod
    def get_final_energy(iso_dirname: Path) -> tuple[float, float]:
        """
        Gets energy and gradient of the molecule

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files

        Returns
        -------
        tuple[float, float]
            energy of the molecule, gradient of the energy
        """

        final_file = iso_dirname / "xtbopt.xyz"
        energy = np.nan
        gradient = np.nan
        with open(final_file, "r") as f:
            for line in f.readlines():
                if "energy:" in line:
                    energy_match = re.search(Xtb._ENERGY_REGEX, line)
                    gradient_match = re.search(Xtb._GRADIENT_REGEX, line)
                    if energy_match and gradient_match:
                        energy = float(energy_match.group(1))
                        gradient = float(gradient_match.group(1))
                        break
        return (energy, gradient)

    @staticmethod
    def get_trajectory(iso_dirname: Path, filename: str) -> list[ConfTag]:
        """
        Gets energy and gradient of the molecule

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files
        filename
            Path of output file with trajectory

        Returns
        -------
        List[ConfTag]
           list of dictionaries contaning results about single conformers and about
           individual atoms in the conformer
        """
        trajectory_file = iso_dirname / filename
        with open(trajectory_file, "r") as f:
            read_coords = 0
            conformers: list[ConfTag] = []
            conformer_energy = np.nan
            gradient = np.nan
            number_atoms = 0
            atoms_list: list[AtomEntry] = []
            atom_id = int(1)
            for line in f.readlines():
                line_lst = line.split()
                if read_coords == 1 and line_lst[0] == "energy:" and line_lst[2] == "gnorm:":
                    conformer_energy = float(line_lst[1])
                    gradient = float(line_lst[3])
                elif read_coords == 1 and line_lst[0] in atom_string:
                    
                    atom_entry: AtomEntry = {'element': str(line_lst[0]),
                            'atom_id': atom_id,
                            'coords': [float(line_lst[1]), float(line_lst[2]), float(line_lst[3])]
                    }
                    
                    atoms_list.append(atom_entry)
                    atom_id += 1
                elif len(line_lst) == 1:
                    if number_atoms != 0:
                        conf_tag: ConfTag = {'atoms': atoms_list, 'energy': conformer_energy, 'gradient': gradient}
                        conformers.append(conf_tag)
                    number_atoms = int(line_lst[0])
                    atom_id = 1
                    read_coords = 1
                    atoms_list = []
            
            conf_tag_last: ConfTag = {'atoms': atoms_list, 'energy': conformer_energy, 'gradient': gradient}
            conformers.append(conf_tag_last)

        if not conformers:
            raise ValueError("Trajectory not found in molecule")
        return conformers

    @staticmethod
    def get_xyz_json(iso_dirname: Path) -> list[AtomEntry]:
        """
        Gets atomic information as list of dictionaries

        Parameters
        ----------
        iso_dirname
            Path of the directory containing the calculation files


        Returns
        -------
        list[dict[str, str | int | list[float]]]
           list of dictionaries contaning results about individual atoms in the conformer
        """

        trajectory_file = iso_dirname / "input.sdf"
        sdf_loader = Loader(str(trajectory_file))
        molecule = sdf_loader.molecule()

        atoms_record: list[AtomEntry] = []
        for atom in molecule.atoms:
            atom_entry: AtomEntry = {'element': atom.label, 'atom_id': atom.number, 'coords': atom.position}
            atoms_record.append(atom_entry)

        return atoms_record

    def _parse_xtb_outputs(
        self,
        mols: list[IsomerCollection],
        mol_outputs: list[list[list[Path]]],
        results: list[CompletedProcess[bytes]],
    ) -> None:
        """
        Parse xtb outputs

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

        iso_dict = {}
        for mol in mols:
            for iso in mol.molecules:
                iso_dict[iso.get_tag("xtb_iso_idx")] = iso

        if self.fold.is_set:
            if not os.path.exists(self.fold.value):
                os.mkdir(self.fold.value)
        else:
            self.logger.info("Not saving calculation files")

        count = 0
        for i, mol_folder in enumerate(mol_outputs):
            mol_path = self.fold.value / f"mol-{i}"
            if os.path.exists(mol_path):
                shutil.rmtree(mol_path)
            os.mkdir(mol_path)

            for j, iso_dirname in enumerate(mol_folder):
                isomer = iso_dict[f"{i}_{j}"]
                isomer_final_geometries: dict[int, list[ConfTag] | list[AtomEntry]] = {}
                isomer_trajectories: dict[int, list[ConfTag] | str] = {}
                isomer_charges = {}
                isomer_wbos = {}
                isomer_gradients = {}
                isomer_energies = {}
                isomer_xtb_exit_codes = {}

                isomer_path = mol_path / f"{isomer.get_tag('component')}"
                if os.path.exists(isomer_path):
                    shutil.rmtree(isomer_path)
                os.mkdir(isomer_path)

                for k, conf_name in enumerate(iso_dirname):
                    conformer = isomer.conformers[k]
                    conf_output = conf_name / "xtbopt.xyz"
                    conf_stdout = results[count].stdout.decode()

                    with open(conf_name / f"{isomer.get_tag('component')}_conf{k}_xtb_out.txt", "w") as out:
                        out.write(conf_stdout)

                    if not conf_output.exists() and self.opt.value:
                        self.logger.warning("XTB optimisation failed for '%s'", conformer)
                        continue

                    ### check calculation status
                    exit_code = 1
                    pattern = "convergence criteria satisfied after"
                    for line in conf_stdout.split("\n"):
                        if pattern in line:
                            exit_code = 0
                        else:
                            continue

                    isomer_xtb_exit_codes[k] = exit_code

                    try:
                        isomer_charges[k] = Xtb.get_atomic_charges(conf_name)
                        isomer_wbos[k] = Xtb.get_wieberg_bo(conf_name)
                    except ValueError:
                        log.info(f"charges and wbo not available for conformer {k} {conf_name}")

                    if self.opt.value:
                        isomer_trajectories[k] = Xtb.get_trajectory(conf_name, "xtbopt.log")
                        isomer_final_geometries[k] = Xtb.get_trajectory(conf_name, "xtbopt.xyz")
                        isomer_energies[k] = Xtb.get_final_energy(conf_name)[0]
                        isomer_gradients[k] = Xtb.get_final_energy(conf_name)[1]

                    else:
                        isomer_energies[k] = _xtb_energy_parser_sp(conf_stdout)
                        isomer_gradients[k] = _xtb_gradient_parser_sp(conf_stdout)
                        isomer_final_geometries[k] = Xtb.get_xyz_json(conf_name)
                        isomer_trajectories[k] = json.dumps(
                            [
                                {
                                    "atoms": isomer_final_geometries[k],
                                    "energy": isomer_energies[k],
                                    "gradient": isomer_gradients[k],
                                }
                            ]
                        )

                    if os.path.exists(isomer_path / f"conf{k}"):
                        shutil.rmtree(isomer_path / f"conf{k}")
                    shutil.copytree(conf_name, isomer_path / f"conf{k}")
                    count += 1

                isomer.set_tag("XTB_exit_codes", json.dumps(isomer_xtb_exit_codes))
                isomer.set_tag("XTB_geometries", json.dumps(isomer_final_geometries))
                isomer.set_tag("xtb_energy", json.dumps(isomer_energies))
                
    def run(self) -> None:
        mols = self.inp.receive()

        commands: list[str] = []
        confs_paths: list[Path] = []
        mol_outputs: list[list[list[Path]]] = []

        for i, mol in enumerate(mols):
            mol_path = Path(f"mol-{i}").absolute()
            mol_path.mkdir()
            isomer_outputs: list[list[Path]] = []
            self.logger.info("XTB optimisation for molecule %s: '%s'", i, mol)

            for j, isomer in enumerate(mol.molecules):
                self.logger.info("  XTB optimisation for isomer %s: '%s'", j, isomer)
                isomer.set_tag("xtb_iso_idx", f"{i}_{j}")
                iso_path = mol_path / f"isomer-{j}"
                iso_path.mkdir()
                conformer_outputs: list[Path] = []
                conformer_tag_dict = {}

                for k, conformer in enumerate(isomer.conformers):
                    self.logger.info("XTB optimisation for conformer %s: '%s'", k, conformer)
                    conformer_tag_dict[k] = f"{i}_{j}_{k}"
                    conf_path = iso_path / f"conformer-{k}"
                    conf_path.mkdir()
                    confs_paths.append(conf_path)
                    input_flname = f"{isomer.get_tag('component')}_conf{k}_inp.xyz"
                    input_path = conf_path / input_flname
                    output_dirname = conf_path

                    if (
                        isomer.has_tag("parameters")
                        and len(cast(list[Any], isomer.get_tag("parameters"))) > 1
                    ):
                        isomer_charge = int(cast(list[Any], isomer.get_tag("parameters"))[0])
                        isomer_mult = int(cast(list[Any], isomer.get_tag("parameters"))[1])
                    else:
                        isomer_charge = self.charge.value
                        isomer_mult = self.multiplicity.value

                    keywords = (
                        f"--T {self.n_threads_job.value} "
                        f"-c {str(int(isomer_charge))} -u {str(int(isomer_mult - 1))} "
                        f"--gbsa {self.solvent.value}")

                    if isomer.has_tag("constraints"):
                        constraints = "--input " + create_constraints_xtb(isomer, conf_path)
                        keywords += f" {constraints} "
                        self.logger.info(f"found constraint {constraints} for isomer {j}")
                    else:
                        constraints = ""
                        self.logger.info(f"no constraint for isomer {j}")

                    try:
                        conformer.to_xyz(path=input_path, tag_name=f"{i}_{j}_{k}")
                    except ValueError as err:
                        self.logger.warning(
                            "Skipping '%s' due to XYZ conversion error:\n %s", conformer, err
                        )

                    if self.opt.value:
                        keywords += " --opt"
                    command = f"{self.runnable['xtb']} {input_path} {keywords}"
                    commands.append(command)
                    conformer_outputs.append(output_dirname)
                isomer_outputs.append(conformer_outputs)

            mol_outputs.append(isomer_outputs)

        self.logger.info(f"Commands before run_multi: {commands}")

        # Run all commands at once
        results = self.run_multi(
            commands,
            working_dirs=confs_paths,
            verbose=False,
            raise_on_failure=True,
            n_jobs=self.n_jobs.value
        )

        # Convert each pose to SDF, update isomer conformation
        self._parse_xtb_outputs(mols, mol_outputs, results)
        self.out.send(mols)


@pytest.fixture
def testing() -> list[str]:
    return ["CNC(=O)", "CCO"]


class TestSuiteXtb:
    @pytest.mark.needs_node("xtb")
    def test_Xtb(
        self,
        temp_working_dir: Any,
        test_config: Any,
    ) -> None:
        rig = TestRig(Xtb, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in ["C", "N"]]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs]})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 2
        for mol in mols:
            assert np.isfinite(mol.molecules[0].tags["xtb_energy"])
            assert np.isfinite(mol.molecules[0].tags["xtb_gradient"])
            assert len(json.loads(mol.molecules[0].tags["xtb_trajectory"])) > 1
        assert len(json.loads(mols[0].molecules[0].tags["xtb_charges"])) == 5
        assert len(json.loads(mols[0].molecules[0].tags["xtb_wbos"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_charges"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_wbos"])) == 3

    @pytest.mark.needs_node("xtb")
    @pytest.mark.needs_node("crest")
    def test_Crest_Xtb(
        self,
        testing: Any,
        test_config: Any,
    ) -> None:
        flow = Workflow(name="test_xtb", level="INFO", cleanup_temp=False)
        flow.config = test_config

        load = flow.add(LoadData[list[str]])
        embe = flow.add(Smiles2Molecules)
        crest_nod = flow.add(Crest)
        opt = flow.add(Xtb)
        ret = flow.add(Return[list[IsomerCollection]])

        flow.connect_all(
            (load.out, embe.inp),
            (embe.out, crest_nod.inp),
            (crest_nod.out, opt.inp),
            (opt.out, ret.inp),
        )

        load.data.set(testing)
        embe.n_variants.set(1)

        flow.check()
        flow.execute()

        mols = ret.get()

        assert mols is not None
        assert len(mols) == 2
        for mol in mols:
            assert np.isfinite(mol.molecules[0].tags["xtb_energy"])
            assert np.isfinite(mol.molecules[0].tags["xtb_gradient"])
            assert len(json.loads(mol.molecules[0].tags["xtb_trajectory"])) > 1
        assert len(json.loads(mols[0].molecules[0].tags["xtb_charges"])) == 5
        assert len(json.loads(mols[0].molecules[0].tags["xtb_wbos"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_charges"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_wbos"])) == 3

    @pytest.mark.needs_node("xtb")
    def test_Xtb_SP(
        self,
        temp_working_dir: Any,
        test_config: Any,
    ) -> None:
        rig = TestRig(Xtb, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in ["C", "N"]]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs]}, parameters={"opt": False})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 2
        for mol in mols:
            assert len(mol.molecules[0].tags["xtb_energy"]) > 1
            assert len(mol.molecules[0].tags["xtb_gradient"]) > 1
            assert len(json.loads(mol.molecules[0].tags["xtb_trajectory"])) == 1
        assert len(json.loads(mols[0].molecules[0].tags["xtb_charges"])) == 5
        assert len(json.loads(mols[0].molecules[0].tags["xtb_wbos"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_charges"])) == 4
        assert len(json.loads(mols[1].molecules[0].tags["xtb_wbos"])) == 3
