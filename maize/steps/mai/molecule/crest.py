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
import pickle
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
    isomer_name = isomer.get_tag('logging_name')
    suffix = 'xyz' if isomer.has_tag('connectivity') else 'sdf'

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
    tags = {"chemistry", "semiempirical", "scorer", "sampler"}

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

    charge: Parameter[int] = Parameter(default=0)
    """Charge of the molecule"""

    multiplicity: Parameter[int] = Parameter(default=1)
    """Multiplicity of the molecule"""

    name: Parameter[str] = Parameter(default="Crest_calculation")
    """Name for the job to be used in the logging."""
    
    solvent: Parameter[str] = Parameter(default="ether")
    """Solvent for the alpb implicit solvation model"""

    scratch: Parameter[str] = Parameter(default="/projects/mai/projects/reactivity_preds/test_maize/local_scratch/crest_scratch")
    """Scratch folder for calculation"""

    def _parse_crest_outputs(
        self,
        isomercollection_list: list[IsomerCollection],
        isomer_outputs: list[list[Path]],
        results: list[CompletedProcess[bytes]],
    ) -> None:
        """
        Parses crest output

        Parameters
        ----------
        mols
            List of IsomerCollection objects corresponding to the molecules in
            the calculation
        isomer_outputs
            list containing list of paths for individual calculation output files
        results
            Results of the jobs
        """

        isomer_dict = {isomer.get_tag("crest_idx"): isomer for isomercollection in isomercollection_list for isomer in isomercollection.molecules}
        count = 0

        if not os.path.exists(self.fold.value):
            os.mkdir(self.fold.value)

        for i, isomer_folder in enumerate(isomer_outputs):
            component_path = self.fold.value / f"mol-{i}"
            if os.path.exists(component_path):
                shutil.rmtree(component_path)
            os.mkdir(component_path)

            for j, isomer_dirname in enumerate(isomer_folder):
                isomer = isomer_dict[f"{i}_{j}"]
                isomer.set_tag("crest_exit_code", 1)
                self.logger.info(isomer.get_tag('logging_name'))

                isomer_output = isomer_dirname / "crest_conformers.xyz"
                isomer_stdout = results[count].stdout.decode()

                with open(isomer_dirname / f"{isomer.get_tag('logging_name')}_crest_out.txt", "w") as out:
                    out.write(isomer_stdout)

                if not isomer_output.exists():
                    self.logger.warning("Crest failed for '%s'", isomer.get_tag('logging_name'))
                    continue

                isomer = update_conformers_from_xyz(isomer, isomer_output)

                # loop through the standard output and search for normal crest termination.
                pattern = "CREST terminated normally."
                pattern_found = False
                for line in isomer_stdout.split("\n"):
                    if pattern in line:
                        pattern_found = True
                        isomer.set_tag("crest_exit_code", 0)

                try:
                    energy_best = isomer.get_tag("crest_energies")
                except KeyError:
                    energy_best = f"error in loading the energy for {isomer_output}"
                self.logger.info(energy_best)
                shutil.copytree(isomer_dirname, component_path / f"{isomer.get_tag('logging_name')}")

                count += 1

    def run(self) -> None:
        isomercollection_list = self.inp.receive()
        commands: list[str] = []
        isomer_paths: list[Path] = []
        molecule_outputs: list[list[Path]] = []

        for i, mol in enumerate(isomercollection_list):
            self.logger.info(f"Looping through isomercollection_list: {i}")

            try:
                logging_name: str = str(mol.molecules[0].get_tag('component'))
            except:
                logging_name = self.name.value
            
            component_path = Path(f"mol-{i}").absolute()
            self.logger.debug(f" this is the path received by CREST {component_path}")
            component_path.mkdir()
            isomer_outputs: list[Path] = []
            self.logger.info(f"CREST calculations for molecule {i}: {logging_name}")

            for j, isomer in enumerate(mol.molecules):
                self.logger.info(f"Looping through mol.molecules: {j}")

                self.logger.info(f"{logging_name}: CREST simulation for isomer {j}: {isomer}")
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

                isomer_path = component_path / f"isomer-{j}"
                isomer_path.mkdir()
                isomer_paths.append(isomer_path)
                
                # check for constraints to apply to the calculations
                if isomer.has_tag("constraints"):
                    constraints = "--cinp " + create_constraints_crest(isomer, isomer_path) + " --subrmsd"
                    self.logger.info(f"found constraint {constraints} for isomer {j}")
                else:
                    constraints = ""
                output_dirname = isomer_path

                # writes sdf input file for the crest calculation
                if isomer.has_tag('connectivity'):
                    if isomer.has_tag('g16_mm_geometries'):
                        self.logger.info(f"loading serialised geom from UFF optimisation on gaussian.")
                        json_string = json.loads(cast(str, isomer.get_tag('g16_mm_geometries')))
                        mm_json = json_string[str(j)]
                        mm_iso = Structure(
                            [EntryCoord(element=ec['element'], coords=ec['coords']).to_Atom() for ec in mm_json]
                            )
                        input_flname = f"{isomer.get_tag('logging_name')}_crest_inp.xyz"
                        input_path = isomer_path / input_flname
                        mm_iso.write_xyz(input_path)

                    elif isomer.has_tag('gfnff_geometries'):
                        self.logger.info(f"loading serialised geom from GFN-FF optimisation on xtb.")
                        json_string = json.loads(cast(str, isomer.get_tag('gfnff_geometries')))
                        gfnff_json = json_string[str(j)]
                        gfnff_iso = Loader.molecule_from_json(gfnff_json, f'isomer-{j}')
                        input_flname = f"{isomer.get_tag('logging_name')}_crest_inp.xyz"
                        input_path = isomer_path / input_flname
                        gfnff_iso.write_xyz(input_path)
                    else:
                        input_flname = f"{isomer.get_tag('logging_name')}_crest_inp.sdf"
                        input_path = isomer_path / input_flname
                        isomer.to_sdf(path=input_path)
                
                else:               
                    input_flname = f"{isomer.get_tag('logging_name')}_crest_inp.sdf"
                    input_path = isomer_path / input_flname
                    isomer.to_sdf(path=input_path)
                

                # writes keywords for specific commands for the
                # calculation related to the compound properties
                keywords = (
                    f"--{self.method.value} -T {self.n_threads_job.value} "
                    f"-c {str(int(isomer_charge))} -u {str(int(isomer_mult - 1))} --alpb {str(self.solvent.value)} --squick --ewin 3.0 --ethr 0.15"
                )
                if isomer.has_tag("constraints"):
                    keywords += f" --noreftopo --nci"
                command = f"{self.runnable['crest']} {input_path} {keywords}"
                commands.append(command)
                self.logger.info(command)

                isomer_outputs.append(output_dirname)

            molecule_outputs.append(isomer_outputs)
            

        self.logger.info(f"Commands before run_multi: {commands}")

        # Run all commands at once
        results = self.run_multi(
            commands,
            working_dirs=isomer_paths,
            verbose=True,
            raise_on_failure=False,
            n_jobs=self.n_jobs.value,
        )

        self._parse_crest_outputs(isomercollection_list, molecule_outputs, results)
        self.out.send(isomercollection_list)

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
