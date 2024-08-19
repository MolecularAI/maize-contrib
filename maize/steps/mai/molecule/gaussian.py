from pathlib import Path
from typing import Callable, Any, List, Dict, cast
from subprocess import CompletedProcess
import dataclasses
import logging
import json
import pytest
import os
import shutil

from maize.core.node import Node
from maize.core.workflow import Workflow
from maize.core.interface import Input, Output, Parameter, Flag, FileParameter
from maize.utilities.execution import JobResourceConfig
from maize.utilities.testing import TestRig
from maize.steps.io import LoadData, Return

from maize.utilities.chem import IsomerCollection
from maize.steps.mai.molecule.compchem_utils import Loader, g_Output

log = logging.getLogger("run")


class Gaussian(Node):
    """
    Runs Gaussian16 simulations on IsomerCollection class.

    Currently only single point calculations are allowed


    References
    ----------
    API documentation: http://gaussian.com/
    """

    required_callables = ["gaussian"]

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    fold: FileParameter[Path] = FileParameter(optional=True)
    """path of folder to dump results outputs"""

    mode: Parameter[str] = Parameter(default="sp")
    """type of calculation to perform"""

    memory: Parameter[int] = Parameter(default=16000)
    """ memory to be used by the gaussian job (in MB)"""

    n_proc: Parameter[int] = Parameter(default=12)
    """number of processors to be used by the gaussian job"""

    charge: Parameter[int] = Parameter(default=0)
    """Structure formal charge"""

    multiplicity: Parameter[int] = Parameter(default=1)
    """Number of unpaired electons in the structure"""

    batch: Flag = Flag(default=True)
    """Flag to submit to SLURM queueing system"""

    n_jobs: Parameter[int] = Parameter(default=8)
    """Number of parallel processes to use"""

    n_threads_job: Parameter[int] = Parameter(default=1)
    """Number of parallel processes to use"""

    solvent: Parameter[str] = Parameter(default="acetonitrile")
    """Type of solvent for implicit solvation"""

    def _parse_gaussian_outputs(
        self,
        mols: list[IsomerCollection],
        mol_outputs: list[list[list[Path]]],
        results: list[CompletedProcess[bytes]],
    ) -> None:
        """
        Parses gaussian output

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
                iso_dict[iso.get_tag("g16_iso_idx")] = iso

        if self.fold.is_set:
            if not os.path.exists(self.fold.value):
                os.mkdir(self.fold.value)
        else:
            self.logger.info("Not saving calculation files.")

        count = 0
        for i, mol_folder in enumerate(mol_outputs):
            mol_path = self.fold.value / f"mol-{i}"
            if os.path.exists(mol_path):
                shutil.rmtree(mol_path)
            os.mkdir(mol_path)

            for j, iso_dirname in enumerate(mol_folder):
                isomer = iso_dict[f"{i}_{j}"]
                isomer_g16_geometries = {}
                isomer_g16_energies = {}
                isomer_g16_exit_codes = {}

                isomer_path = mol_path / f"{isomer.get_tag('component')}"
                if os.path.exists(isomer_path):
                    shutil.rmtree(isomer_path)
                os.mkdir(isomer_path)

                for k, conf_name in enumerate(iso_dirname):
                    conformer = isomer.conformers[k]
                    conf_output = g_Output(str(conf_name / f"input.log"))
                    conf_stdout = results[count].stdout.decode()

                    # check termination of the calculation
                    exit_code = 1
                    if conf_output.normal_termination():
                        exit_code = 0
                    else:
                        self.logger.warning("Gaussian failed for '%s'", conformer)
                        continue

                    isomer_g16_exit_codes[k] = exit_code

                    result_output = conf_output.intermediate_output()
                    if self.mode.value == "sp":
                        try:
                            conf_geometry = result_output.mol_dict
                            serialised_geom = [dataclasses.asdict(a) for a in conf_geometry]
                            isomer_g16_geometries[k] = serialised_geom

                        except:
                            log.info(
                                f"energies and geometries not available for conformer {k} {conf_name}"
                            )
                        isomer_g16_energies[k] = result_output.energy

                    if os.path.exists(isomer_path / f"conf{k}"):
                        shutil.rmtree(isomer_path / f"conf{k}")
                    shutil.copytree(conf_name, isomer_path / f"conf{k}")
                    count += 1

                isomer.set_tag("g16_exit_codes", json.dumps(isomer_g16_exit_codes))
                isomer.set_tag("final_geometries", json.dumps(isomer_g16_geometries))
                isomer.set_tag("g16_energy", json.dumps(isomer_g16_energies))

    def run(self) -> None:
        oniom = "N/A"
        mode_calc = str(self.mode.value)

        mols = self.inp.receive()

        commands: list[str] = []
        confs_paths: list[Path] = []
        mol_outputs: list[list[list[Path]]] = []

        for i, mol in enumerate(mols):
            mol_path = Path(f"mol-{i}")
            mol_path.mkdir()
            isomer_outputs: list[list[Path]] = []
            self.logger.info("Gaussian optimisation for molecule %s: '%s'", i, mol)

            for j, isomer in enumerate(mol.molecules):
                self.logger.info("Gaussian optimisation for isomer %s: '%s'", j, isomer)
                isomer.set_tag("g16_iso_idx", f"{i}_{j}")

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

                conf_geometries_tag = json.loads(cast(str, isomer.get_tag("XTB_geometries")))
                conformer_outputs: list[Path] = []
                conformer_tag_dict = {}

                for k in range(len(conf_geometries_tag)):
                    self.logger.info("Gaussian Single Point for conformer %s", k)
                    conformer_tag_dict[k] = f"{i}_{j}_{k}"
                    conf_json = conf_geometries_tag[str(k)]
                    self.logger.info(conf_json)
                    conformer = Loader.molecule_from_json(conf_json, f"conformer-{k}")
                    conf_path = iso_path / f"conformer-{k}"
                    conf_path.mkdir()
                    confs_paths.append(conf_path)
                    input_flname = "input.xyz"
                    output_dirname = conf_path
                    if (
                        isomer.has_tag("constraints")
                        and len(cast(list[Any], isomer.get_tag("constraints"))) > 0
                    ):
                        # create function for gaussian costraints if we want them
                        constraints = ""
                        self.logger.info(
                            f"found constraint for isomer {j} but I won't do nothing for now"
                        )
                    else:
                        constraints = ""
                        self.logger.info(f"no constraint for isomer {j}")

                    try:
                        conformer.write_xyz(conf_path / input_flname)

                    except ValueError as err:
                        self.logger.warning(
                            "Skipping '%s' due to XYZ conversion error:\n %s", conformer, err
                        )
                    link = False
                    loader_xyz = Loader(str(conf_path / input_flname))
                    conformer_mk = loader_xyz.molecule_xyz()
                    conf_g16_inp = conformer_mk.export_g16_into_maize(
                        isomer_charge,
                        isomer_mult,
                        str(self.solvent.value),
                        str(self.memory.value),
                        str(self.n_proc.value),
                        mode_calc,
                        oniom,
                        link,
                    )

                    command = f"{self.runnable['gaussian']} {conf_g16_inp} "
                    commands.append(command)
                    conformer_outputs.append(output_dirname)
                isomer_outputs.append(conformer_outputs)

            mol_outputs.append(isomer_outputs)

        # Run all commands at once
        results = self.run_multi(
            commands,
            working_dirs=confs_paths,
            verbose=False,
            raise_on_failure=False,
            n_jobs=self.n_jobs.value
        )
        # Convert each pose to SDF, update isomer conformation
        self._parse_gaussian_outputs(mols, mol_outputs, results)
        for mol in mols:
            for iso in mol.molecules:
                self.logger.info(iso.tags)
        self.out.send(mols)


class TestSuiteGaussian:
    @pytest.mark.needs_node("gaussian")
    def test_g16_SP(
        self,
        temp_working_dir: Any,
        test_config: Any,
    ) -> None:
        rig = TestRig(Gaussian, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in ["CC", "CO"]]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs]})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 2
        for mol in mols:
            for i in range(len(mol.molecules)):
                index = str(i)
                exit_codes = json.loads(mol.molecules[i].tags["g16_exit_codes"])
                final_geoms = json.loads(mol.molecules[i].tags["final_geometries"])
                energies = json.loads(mol.molecules[i].tags["g16_energy"])

                assert exit_codes[index] == 0
                assert final_geoms[index]
                assert energies[index]
