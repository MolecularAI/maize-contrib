from pathlib import Path
from typing import Any, cast
from subprocess import CompletedProcess
from rdkit import Chem
import json
import shutil
import os
import numpy as np
import logging
import pytest

from maize.utilities.testing import TestRig

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Flag, FileParameter

from maize.utilities.chem import Isomer, IsomerCollection, Conformer



log = logging.getLogger("run")


def create_constraints_crest(isomer: Isomer, path: Path) -> str:
    """
    Create constraint file for CREST calculations starting from the isomer
    object and the constrained tag associated. Need to specify the constrained
    and unconstrained files, as well as a reference geometry. For reference see:
    https://crest-lab.github.io/crest-docs/page/examples/example_4.html

    """
    constr_indexes = json.loads(isomer.tags["constraints"])
    isomer_name = isomer.get_tag('component')

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
        f.write("{}\n".format("   force constant=1.0 "))
        f.write("{}\n".format(f"   reference={isomer_name}_crest_inp.sdf"))
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
                            log.warning("Unable to parse conformer, error: %s", err)
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

    method: Parameter[str] = Parameter(default="--mquick")
    """Accuracy/speed of the simulation"""

    fold: FileParameter[Path] = FileParameter(optional=True)
    """path of folder to dump results outputs"""

    batch: Flag = Flag(default=True)
    """Flag to submit to SLURM queueing system"""

    n_jobs: Parameter[int] = Parameter(default=100)
    """Number of parallel processes to use"""

    n_threads_job: Parameter[int] = Parameter(default=2)
    """Number of parallel processes to use"""

    charge: Parameter[int] = Parameter(default=0)
    """Charge of the molecule"""

    multiplicity: Parameter[int] = Parameter(default=1)
    """Multiplicity of the molecule"""

    solvent: Parameter[str] = Parameter(default="acetonitrile")
    """Solvent for the gbsa implicit solvation model"""

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
                self.logger.info(isomer.get_tag('component'))

                iso_output = iso_dirname / "crest_conformers.xyz"
                iso_stdout = results[count].stdout.decode()

                with open(iso_dirname / f"{isomer.get_tag('component')}_crest_out.txt", "w") as out:
                    out.write(iso_stdout)

                if not iso_output.exists():
                    self.logger.warning("Crest failed for '%s'", isomer)
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
                shutil.copytree(iso_dirname, mol_path / f"{isomer.get_tag('component')}")

                count += 1

    def run(self) -> None:
        mols = self.inp.receive()
        commands: list[str] = []
        iso_paths: list[Path] = []
        mol_outputs: list[list[Path]] = []

        for i, mol in enumerate(mols):

            mol_path = Path(f"mol-{i}")
            mol_path.mkdir()
            isomer_outputs: list[Path] = []
            self.logger.info("Crest optimisation for molecule %s: '%s'", i, mol)

            for j, isomer in enumerate(mol.molecules):
                self.logger.info("Crest optimisation for isomer %s: '%s'", j, isomer)
                isomer.set_tag("crest_idx", f"{i}_{j}")

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
                input_flname = f"{isomer.get_tag('component')}_crest_inp.sdf"
                input_path = iso_path / input_flname
                # check for constraints to apply to the calculations
                if isomer.has_tag("constraints"):
                    constraints = "--cinp " + create_constraints_crest(isomer, iso_path) + " --subrmsd"
                    self.logger.info(f"found constraint {constraints} for isomer {j}")
                else:
                    constraints = ""
                    self.logger.info(f"no constraint for isomer {j}")
                output_dirname = iso_path

                # writes sdf input file for the crest calculation
                try:
                    isomer.to_sdf(path=input_path)
                except ValueError as err:
                    self.logger.info("Skipping '%s' due to SDF conversion error:\n %s", isomer, err)

                # writes keywords for specific commands for the
                # calculation related to the compound properties
                keywords = (
                    f"{self.method.value} --T {self.n_threads_job.value} "
                    f"-c {str(int(isomer_charge))} -u {str(int(isomer_mult - 1))} "
                    f"--gbsa {self.solvent.value}"
                )
                if isomer.has_tag("constraints"):
                    keywords += f" {constraints}"
                # manually added -noreftopo keyword by bob van schendel
                keywords += " --noreftopo "

                command = f"{self.runnable['crest']} {input_path.absolute()} {keywords}"
                commands.append(command)
                self.logger.info(command)

                isomer_outputs.append(output_dirname)

            mol_outputs.append(isomer_outputs)
            self.logger.info(f"isomer outputs from crest: {isomer_outputs}")

        self.logger.info(f"Commands before run_multi: {commands}")

        # Run all commands at once
        results = self.run_multi(
            commands,
            working_dirs=iso_paths,
            verbose=True,
            raise_on_failure=False,
            n_jobs=self.n_jobs.value,
        )

        # Convert each pose to SDF, update isomer conformation
        self._parse_crest_outputs(mols, mol_outputs, results)
        self.out.send(mols)


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
