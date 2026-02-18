from pathlib import Path
from typing import Any, cast
from numpy.typing import NDArray
from subprocess import CompletedProcess
import dataclasses
import numpy as np
import logging
import json
import pytest
import os
import shutil

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Flag, FileParameter
from maize.utilities.testing import TestRig

from maize.utilities.chem import IsomerCollection
from maize.steps.mai.molecule.compchem_utils import Loader, g_Output, detect_clashes, check_collaps

log = logging.getLogger("run")


def convert_connectivity_tag(strings: list[str], mlength: int) -> str:
    """
    Converts connectivity list format in Isomer tag into the
    right format for gaussian calculations using the geom=connectivity
    keyword.

    Parameters
    ----------
    strings
        list of strings for connectivity contained into isomer tag.
    mlength
        number of atoms in the molecule

    Returns
    -------
    str
        text compatible with gaussian format for connectivity.
    """

    connect = []

    for i in range(0, len(strings), 3):

        first = int(strings[i].replace("(", ""))
        second = int(strings[i + 1])
        third = float(strings[i + 2].replace(")", ""))
        connect.append([first, second, third])

    relationships = {}
    for relationship in connect:
        if relationship[0] not in relationships:
            relationships[relationship[0]] = [[relationship[1], relationship[2]]]
        else:
            relationships[relationship[0]].append([relationship[1], relationship[2]])

    text = ""
    for i in range(1, mlength + 1):
        if i in relationships:
            text += (
                str(i)
                + " "
                + " ".join(str(value) for sublist in relationships[i] for value in sublist)
                + "\n"
            )
        else:
            text += str(i) + "\n"

    return text


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

    mode: Parameter[str] = Parameter(default="dft")
    """Select method to use for calculation (DFT or MM)"""

    job_type: Parameter[str] = Parameter(default="sp")
    """type of job to perform"""

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

    label: Parameter[str] = Parameter(default="Gaussian_calculation")
    """Name of the Gaussian job to be used in the logging."""

    functional: Parameter[str] = Parameter(default="MN15")
    """functional to be used in the calculation"""

    basis_set: Parameter[str] = Parameter(default="cc-pvdz")
    """basis set to be used in the calculation for standard atoms"""

    solvent: Parameter[str] = Parameter(default="acetonitrile")
    """Type of solvent for implicit solvation"""

    link: Flag = Flag(default=False)
    """Ask for a linked job in Gaussian"""

    extra: Parameter[list[str]] = Parameter(default=[])
    """Include additional keywords to gaussian calculations"""

    def _parse_gaussian_outputs(
        self,
        mols: list[IsomerCollection],
        mol_outputs: list[list[list[Path] | Path]],
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
                iso_dict[iso.get_tag(f"g16_{self.mode.value}_iso_idx")] = iso

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
                isomer_g16_outputs = {}
                isomer_g16_free_energies = {}
                isomer_g16_freq_analysis = {}
                isomer_g16_min_energy: dict[str, float] = {"conf-NA": 1000000}
                isomer_ni_distances = {}

                if self.mode.value == "dft":
                    isomer_path = mol_path / f"{isomer.get_tag('logging_name')}"
                    if os.path.exists(isomer_path):
                        shutil.rmtree(isomer_path)
                    os.mkdir(isomer_path)

                    for k, conf_name in enumerate(cast(list[Path], iso_dirname)):
                        try:
                            conf_outpath = str(
                                conf_name / f"{isomer.get_tag('logging_name')}_conf{k}.log"
                            )
                            conf_output = g_Output(conf_outpath)
                            isomer_g16_outputs[k] = conf_outpath

                        except:
                            self.logger.info(
                                f" I could not find the output file for {isomer.get_tag('logging_name')}_conf{k}.log"
                            )
                            raise ValueError("could not find conf output name")

                            # check termination of the calculation
                        exit_code = 1
                        if conf_output.normal_termination():
                            self.logger.info(
                                f"DFT calculation normally terminated for {isomer.get_tag('logging_name')}"
                            )
                            exit_code = 0
                            isomer_g16_exit_codes[k] = exit_code
                            result_output = conf_output.intermediate_output()

                            try:
                                conf_geometry = result_output.structure_to_tag()
                                serialised_geom = [conf_geometry]
                                self.logger.info(
                                    f"this is serialised geometry for conf-{k}:\n {serialised_geom}"
                                )
                                isomer_g16_geometries[k] = serialised_geom

                            except:
                                log.info(
                                    f"energies and geometries not available for conformer {k} {conf_name}"
                                )

                            if result_output.energy:
                                energy = result_output.energy
                                isomer_g16_energies[k] = energy
                                if (
                                    self.job_type.value == "minimum"
                                    or self.job_type.value == "GSH_temp"
                                ):
                                    if conf_output.frequency_analysis_gaussian() == "minimum":
                                        free_energy = float(energy) + float(
                                            conf_output.gibbs_correction()
                                        )
                                        isomer_g16_free_energies[k] = free_energy
                                        isomer_g16_freq_analysis[k] = "Minimum"
                                    else:
                                        free_energy = float(energy) + float(
                                            conf_output.gibbs_correction()
                                        )
                                        isomer_g16_free_energies[k] = free_energy
                                        isomer_g16_freq_analysis[k] = "Saddle point"

                                elif self.job_type.value == "sp" and isomer.has_tag(
                                    "isomer_free_energy_corrections"
                                ):
                                    free_energy = float(energy) + float(
                                        json.loads(
                                            cast(
                                                str,
                                                isomer.get_tag("isomer_free_energy_corrections"),
                                            )
                                        )[str(k)]
                                    )
                                    isomer_g16_free_energies[k] = free_energy

                                if list(isomer_g16_min_energy.values())[0] >= free_energy:
                                    isomer_g16_min_energy = {f"conf-{k}": free_energy}
                                    conf_mol = Loader.molecule_from_json(
                                        serialised_geom, f"conformer-{k}"
                                    )
                                    if (
                                        isomer.has_tag("cc_carbons")
                                        and len(cast(NDArray[np.int32], isomer.get_tag("cc_carbons"))) > 1
                                    ):
                                        if not check_collaps(
                                            conf_mol,
                                            int(cast(NDArray[np.int32], isomer.get_tag("cc_carbons"))[0]),
                                            int(cast(NDArray[np.int32], isomer.get_tag("cc_carbons"))[1]),
                                        ):
                                            if "Ni3_" in cast(str, isomer.get_tag("logging_name")):
                                                bonds_record = {}
                                                for bond in conf_mol.connectivity:
                                                    if (
                                                        bond.atom1.label == "Ni"
                                                        or bond.atom2.label == "Ni"
                                                    ):
                                                        dist = round(bond.length, 2)
                                                        if bond.atom1.label == "C":
                                                            if bond.atom1.aromaticity:
                                                                bonds_record[
                                                                    f"{bond.atom1.label}(Ar)-{bond.atom2.label}"
                                                                ] = dist
                                                            else:
                                                                bonds_record[
                                                                    f"{bond.atom1.label}-{bond.atom2.label}"
                                                                ] = dist

                                                        elif bond.atom2.label == "C":
                                                            if bond.atom2.aromaticity:
                                                                bonds_record[
                                                                    f"{bond.atom1.label}-{bond.atom2.label}(Ar)"
                                                                ] = dist
                                                            else:
                                                                bonds_record[
                                                                    f"{bond.atom1.label}-{bond.atom2.label}"
                                                                ] = dist

                                                        else:
                                                            bonds_record[
                                                                f"{bond.atom1.label}-{bond.atom2.label}"
                                                            ] = dist

                                                isomer_ni_distances = {f"conf-{k}": bonds_record}
                                        else:
                                            isomer_g16_energies[k] = 0.0
                                            isomer_g16_free_energies[k] = 0.0
                                            isomer_g16_geometries.pop(k, None)
                                            self.logger.info(
                                                f"conformer-{k} collapsed to product after optimization. Deleted"
                                            )

                            else:
                                isomer_g16_energies[k] = 0.0
                                isomer_g16_freq_analysis[k] = "error"
                                isomer_g16_free_energies[k] = 0.0

                            if os.path.exists(isomer_path / f"conf{k}"):
                                shutil.rmtree(isomer_path / f"conf{k}")
                            shutil.copytree(conf_name, isomer_path / f"conf{k}")
                            count += 1

                        else:
                            g16_error_message = conf_output.error_check_gaussian()
                            self.logger.info(
                                f"DFT optimization failed: {isomer.get_tag('logging_name')}_conf{k}.log"
                            )
                            self.logger.info(f"Error message: {g16_error_message}")
                            isomer_g16_exit_codes[k] = 1
                            isomer_g16_freq_analysis[k] = f"Error message: {g16_error_message}"
                           
                            count += 1
                    isomer.set_tag("g16_dft_exit_codes", json.dumps(isomer_g16_exit_codes))
                    isomer.set_tag("g16_dft_geometries", json.dumps(isomer_g16_geometries))
                    isomer.set_tag("g16_dft_outputs", json.dumps(isomer_g16_outputs))
                    self.logger.info(
                        f"There are {len(isomer_g16_geometries)} conformers for {isomer.get_tag('logging_name')}"
                    )
                    self.logger.info(
                        f"{isomer.get_tag('logging_name')} -> Conformer geometries: {isomer_g16_geometries}"
                    )

                if self.mode.value == "mm":
                    isomer_g16mm_exit_codes: int = 0
                    isomer_g16mm_geometries: dict[int, list[dict[str, int|float]]] = {}
                    mm_mol_path = cast(Path, iso_dirname)
                    out_name = str(mm_mol_path / f"{isomer.get_tag('logging_name')}_MM_inp.log")
                    mm_output = g_Output(out_name)
                    exit_code = 1
                    self.logger.info(f"Result for {out_name} is: {mm_output.normal_termination()}")
                    if mm_output.normal_termination():
                        exit_code = 0
                        isomer_g16mm_exit_codes = exit_code
                    else:
                        self.logger.warning(
                            "Gaussian MM failed for '%s'", isomer.get_tag("logging_name")
                        )
                        isomer_g16mm_exit_codes = exit_code

                    result_output = mm_output.intermediate_output()
                    if detect_clashes(result_output):
                        self.logger.warning(
                            f"Clashes detected after Gaussian MM run. Potential artifacts for {isomer.get_tag('logging_name')}"
                        )
                        isomer.set_tag("MM_clashes", "Detected")

                    else:
                        self.logger.warning(
                            f"No clashes detected after Gaussian MM run. {isomer.get_tag('logging_name')}"
                        )
                        isomer.set_tag("MM_clashes", "None")

                    try:
                        mm_geometry = result_output.mol_dict
                        mm_serialised_geom = [dataclasses.asdict(a) for a in mm_geometry]
                        isomer_g16mm_geometries[j] = mm_serialised_geom

                    except:
                        log.info(
                            f"energies and geometries not available for conformer {k} {conf_name}"
                        )
                    isomer_path = mol_path / f"{isomer.get_tag('logging_name')}"
                    if os.path.exists(isomer_path):
                        shutil.rmtree(isomer_path)
                    shutil.copytree(cast(Path, iso_dirname), isomer_path)
                    isomer.set_tag("g16_mm_exit_codes", json.dumps(isomer_g16mm_exit_codes))
                    isomer.set_tag("g16_mm_geometries", json.dumps(isomer_g16mm_geometries))

                if isomer_g16_energies:
                    isomer.set_tag("g16_energy", json.dumps(isomer_g16_energies))
                    isomer.set_tag("g16_free_energy", json.dumps(isomer_g16_free_energies))
                    isomer.set_tag("g16_frequency_analysis", json.dumps(isomer_g16_freq_analysis))
                    isomer.set_tag("g16_lowest_conformer", json.dumps(isomer_g16_min_energy))
                    if isomer.get_tag("logging_name") == "Ni3_TrigonalBipyramidal":
                        isomer.set_tag("Ni bond distances", json.dumps(isomer_ni_distances))

    def run(self) -> None:
        mode_calc = str(self.mode.value)
        job_type = str(self.job_type.value)
        others = " ".join(self.extra.value)

        mols = self.inp.receive()

        commands: list[str] = []
        res_paths: list[Path] = []
        mol_outputs: list[list[list[Path] | Path]] = []

        for i, mol in enumerate(mols):
            try:
                logging_name = mol.molecules[0].get_tag("component")
            except:
                logging_name = self.label.value

            mol_path = Path(f"mol-{i}").absolute()
            mol_path.mkdir()
            isomer_outputs: list[list[Path] | Path] = []
            self.logger.info(f"Gaussian calculations for molecule {i}: {logging_name}")

            for j, isomer in enumerate(mol.molecules):
                isomer.set_tag("logging_name", logging_name)
                self.logger.info(
                    f"{mol.molecules[0].get_tag('logging_name')}: Gaussian {mode_calc} {job_type} for isomer {j}: {isomer}"
                )
                isomer.set_tag(f"g16_{mode_calc}_iso_idx", f"{i}_{j}")

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
                
                if mode_calc == "dft":
                    conformer_outputs: list[Path] = []
                    if isomer.has_tag("XTB_geometries"):
                        if isomer.has_tag("Filtered_conformers"):
                            conf_geometries_tag = json.loads(
                                cast(str, isomer.get_tag("Filtered_conformers"))
                            )
                        else:
                            conf_geometries_tag = json.loads(
                                cast(str, isomer.get_tag("XTB_geometries"))
                            )
                        conformer_tag_dict = {}

                        for k in range(len(conf_geometries_tag)):
                            self.logger.info(
                                f"{mol.molecules[0].get_tag('logging_name')} - isomer {j}: Gaussian Single Point for conformer {k}"
                            )
                            conformer_tag_dict[k] = f"{i}_{j}_{k}"
                            conf_json = conf_geometries_tag[str(k)]
                            conformer = Loader.molecule_from_json(conf_json, f"conformer-{k}")
                            conf_path = iso_path / f"conformer-{k}"
                            conf_path.mkdir()
                            res_paths.append(conf_path)
                            input_flname = f"{mol.molecules[0].get_tag('logging_name')}_conf{k}.xyz"
                            output_dirname = conf_path

                            try:
                                conformer.write_xyz(conf_path / input_flname)
                            except ValueError as err:
                                self.logger.warning(
                                    "Skipping '%s' due to XYZ conversion error:\n %s",
                                    conformer,
                                    err,
                                )
                            link = self.link.value
                            conformer.name = str(conf_path / input_flname)[:-4]
                            conf_g16_inp = conformer.export_g16_into_maize(
                                isomer_charge,
                                isomer_mult,
                                str(self.solvent.value),
                                str(self.memory.value),
                                str(self.n_proc.value),
                                job_type,
                                str(self.functional.value),
                                str(self.basis_set.value),
                                others,
                                link,
                            )

                            command = f"{self.runnable['gaussian']} {conf_g16_inp} "
                            commands.append(command)
                            conformer_outputs.append(output_dirname)
                        isomer_outputs.append(conformer_outputs)

                    else:
                        conformer_tag_dict = {}
                        for k, conformation in enumerate(isomer.conformers):
                            self.logger.info(
                                f"{mol.molecules[0].get_tag('logging_name')} - isomer {j}: Gaussian Calc for conformer {k}"
                            )
                            conformer_tag_dict[k] = f"{i}_{j}_{k}"
                            conf_path = iso_path / f"conformer-{k}"
                            conf_path.mkdir()
                            res_paths.append(conf_path)
                            input_flname = f"{isomer.get_tag('logging_name')}_conf{k}.xyz"
                            output_dirname = conf_path

                            try:
                                conf_str = Loader.molecule_from_conformer(conformation)
                                conf_str.write_xyz(conf_path / input_flname)
                            except ValueError as err:
                                self.logger.warning(
                                    "Skipping '%s' due to XYZ conversion error:\n %s",
                                    conformer,
                                    err,
                                )
                            link = False
                            conf_str.name = str(conf_path / input_flname)[:-4]
                            conf_g16_inp = conf_str.export_g16_into_maize(
                                isomer_charge,
                                isomer_mult,
                                str(self.solvent.value),
                                str(self.memory.value),
                                str(self.n_proc.value),
                                job_type,
                                str(self.functional.value),
                                str(self.basis_set.value),
                                others,
                                link,
                            )

                            command = f"{self.runnable['gaussian']} {conf_g16_inp} "
                            commands.append(command)
                            conformer_outputs.append(output_dirname)
                        isomer_outputs.append(conformer_outputs)

                elif mode_calc == "mm":
                    functional = "mm"
                    basis_set = "mm"
                    input_flname = f"{isomer.get_tag('logging_name')}_MM_inp.sdf"
                    input_path = iso_path / input_flname
                    res_paths.append(iso_path)

                    if isomer.has_tag("connectivity"):
                        connectivity = convert_connectivity_tag(
                            cast(list[str], isomer.get_tag("connectivity")),
                            isomer._molecule.GetNumAtoms(),
                        )

                        output_dirname = iso_path

                        try:
                            isomer.to_sdf(path=input_path)
                        except ValueError as err:
                            self.logger.info(
                                "Skipping '%s' due to SDF conversion error:\n %s", isomer, err
                            )

                        iso_struc = Loader.molecule(Loader(str(input_path)))
                        iso_struc.name = str(input_path)[:-4]

                        link = False
                        mm_g16_inp = iso_struc.export_g16_into_maize(
                            isomer_charge,
                            isomer_mult,
                            str(self.solvent.value),
                            str(self.memory.value),
                            str(self.n_proc.value),
                            mode_calc,
                            functional,
                            basis_set,
                            others,
                            link,
                            connectivity,
                        )
                        command = f"{self.runnable['gaussian']} {mm_g16_inp} "
                        commands.append(command)
                        isomer_outputs.append(output_dirname)

                    else:
                        input_path = iso_path / input_flname
                        isomer.to_sdf(path=input_path)

            mol_outputs.append(isomer_outputs)

        # Run all commands at once
        results = self.run_multi(
            commands,
            working_dirs=res_paths,
            verbose=True,
            raise_on_failure=False,
            n_jobs=self.n_jobs.value,
        )
        # Convert each pose to SDF, update isomer conformation

        self._parse_gaussian_outputs(mols, mol_outputs, results)
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