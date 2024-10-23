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


def create_xcontrol_xtb(isomer: Isomer, path: Path, constr: bool = False) -> str:
    """
    Create xcontrol file for xtb calculations starting from the isomer
    object. Xcontrol file allows for additional control on the settings 
    on the calculations, such as constraining options and symmetry control

    Parameters
    ----------
    isomer
        Isomer object corresponding to the molecule in the calculation
    path
        Path of the calculation file.
    constr
        Boolean specifies if constraining options are needed.
    """

    if constr:
        constr_indexes = json.loads(isomer.tags["constraints"])
        isomer_name = isomer.get_tag('component')
        suffix = 'xyz' if isomer.has_tag('connectivity') else 'sdf'

        xcontrol_file = "{}/{}_MM_constraints.inp".format(path, isomer_name)
        unconstrained_indexes = [
            i for i in list(range(1, isomer.n_atoms + 1)) if i not in constr_indexes
        ]

        ## return list of uncostrained indexes in the correct format
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

  
        with open(xcontrol_file, "w") as f:
            f.write("{}\n".format("$constrain"))
            f.write("{}\n".format("   atoms: " + str(constr_indexes).replace("[", "").replace("]", "")))
            f.write("{}\n".format("   force constant=1.0 "))
            f.write("{}\n".format(f"   reference={isomer_name}_MM_inp.{suffix}"))
            f.write("{}\n".format("$metadyn:"))
            f.write("{}\n".format("   atoms: " + uncostrained_string))
            f.write("{}\n".format("$symmetry:"))
            f.write("{}\n".format("   maxat=0"))
            f.write("{}\n".format("$end"))

    else:
        isomer_name = isomer.get_tag('component')
        xcontrol_file = "{}/{}_control.inp".format(path, isomer_name)
        with open(xcontrol_file, "w") as f:
            f.write("{}\n".format("$symmetry:"))
            f.write("{}\n".format("   maxat=0"))
            f.write("{}\n".format("$end"))

    return xcontrol_file


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

    Data for, during and after processing is stored in the following tags:
        * component : string
            name of the component with regards to the reaction mechanism its used in
        * connectivity : list[tuple[int,int,float]]
            column matrix representing the connectivity of the molecule
        * xtb_iso_idx : int
            the index of the isomer in the given isomercollection
        * XTB_exit_codes : str
            JSON exit codes of XTB
        * XTB_geometries : str
            JSON describing the XYZ geometries
        * xtb_energy : str
            JSON describing the energies per conformer
        * xtb_free_energy : str
            JSON describing the free energies per conformer
        * isomer_free_energy_corrections : str
            JSON describing the free energy corrections
        * xtb_lowest_conformer : str
            JSON describing the single conformer with the lowest energy out of all those stores in XTB_geometries

    References
    ----------
    API documentation: https://xtb-docs.readthedocs.io/en/latest/contents.html
    Key citation reference for the XTB methods and current implementation:
    {C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
    S. Spicher, S. Grimme WIREs Comput. Mol. Sci., 2020, 11, e01493. DOI: 10.1002/wcms.1493}

    """
    tags = {"chemistry", "semiempirical", "scorer", "sampler"}

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

    solvent: Parameter[str] = Parameter(default="ether")
    """Solvent for the gbsa implicit solvation model"""

    fold: FileParameter[Path] = FileParameter(optional=True)
    """path of folder to dump results outputs"""

    mode: Parameter[str] = Parameter(default='sqm')

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
    def get_free_energy(output_file: Path) -> tuple[float, float]:
        """
        Gets gibbs free energy and free energy correction for the molecule,
        from the XTB calculation. Correction is made of zero point energy and 
        G(RRHO) contribution.

        Parameters
        ----------
        output_file
            Path of the xtb calculation output file.

        Returns
        -------
        tuple[float, float]
            gibbs free energy, free energy correction
        """

        gibbs_free_energy = np.nan
        gibbs_correction = np.nan
        with open(output_file, "r") as out:
            for line in out:
                if 'FREE ENERGY' in line:
                    energy = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    gibbs_free_energy = float(energy[0])
            
                if 'G(RRHO) contrib.' in line:
                    correction = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    gibbs_correction = float(correction[0]) 

        return (gibbs_free_energy, gibbs_correction)

    @staticmethod
    def check_imaginary(output_file: Path) -> bool:
        """
        Checks for imaginary frequency in the xtb output by searching for
        the specific text pattern.

        Parameters
        ----------
        output_file
          Path of the xtb calculation output file.
        
        Returns
        -------
        bool
            Boolean for the presence of the text pattern.
        """
        with open(output_file, 'r') as file:
            for line in file:
                if 'significant imaginary frequency' in line:
                    return True
        return False
        
    @staticmethod
    def get_trajectory(output_file: Path) -> list[ConfTag]:
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
        trajectory_file = output_file
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
        mol_outputs: list[list[list[Path] | Path]],
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
                

                if self.mode.value == 'sqm':
                    isomer_final_geometries: dict[int, list[ConfTag] | list[AtomEntry]] = {}
                    isomer_trajectories: dict[int, list[ConfTag] | str] = {}
                    isomer_charges = {}
                    isomer_wbos = {}
                    isomer_gradients = {}
                    isomer_energies = {}
                    isomer_free_energies = {}
                    isomer_free_energy_corrections = {}
                    isomer_xtb_exit_codes = {}
                    isomer_path = mol_path / f"{isomer.get_tag('component')}"
                    isomer_min_energy: dict[str, float] = {'conf-NA': 100000}

                    if os.path.exists(isomer_path):
                        shutil.rmtree(isomer_path)
                    os.mkdir(isomer_path)

                    for k, conf_name in enumerate(cast(list[Path], iso_dirname)):
                        conformer = isomer.conformers[k]
                        conf_output = conf_name / "xtbopt.xyz"
                        conf_stdout = results[count].stdout.decode()
                        outfile = conf_name / f"{isomer.get_tag('component')}_conf{k}_xtb_out.txt"
                        with open(outfile, "w") as out:
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
                
                        try:
                            isomer_charges[k] = Xtb.get_atomic_charges(conf_name)
                            isomer_wbos[k] = Xtb.get_wieberg_bo(conf_name)
                        except ValueError:
                            log.info(f"charges and wbo not available for conformer {k} {conf_name}")

                        if self.opt.value:

                            if Xtb.check_imaginary(outfile):
                                self.logger.info(f'Found Imaginary frequency for conf-{k}. Storing distorted geometry for restart.')
                                exit_code = 1
                                conf_free_energy = Xtb.get_free_energy(outfile)[0]
                                isomer_trajectories[k] = Xtb.get_trajectory(conf_name / "xtbopt.log")
                                isomer_final_geometries[k] = Xtb.get_trajectory(conf_name / "xtbhess.xyz")
                                isomer_energies[k] = Xtb.get_final_energy(conf_name)[0]
                                isomer_gradients[k] = Xtb.get_final_energy(conf_name)[1]
                                isomer_free_energies[k] = conf_free_energy
                                isomer_free_energy_corrections[k] = Xtb.get_free_energy(outfile)[1]

                                

                   
                            else:
                                conf_free_energy = Xtb.get_free_energy(outfile)[0]
                                isomer_trajectories[k] = Xtb.get_trajectory(conf_name / "xtbopt.log")
                                isomer_final_geometries[k] = Xtb.get_trajectory(conf_name / "xtbopt.xyz")
                                isomer_energies[k] = Xtb.get_final_energy(conf_name)[0]
                                isomer_gradients[k] = Xtb.get_final_energy(conf_name)[1]
                                isomer_free_energies[k] = conf_free_energy
                                isomer_free_energy_corrections[k] = Xtb.get_free_energy(outfile)[1]
                                if list(isomer_min_energy.values())[0] >= conf_free_energy:
                                    isomer_min_energy = {f'conf-{k}': conf_free_energy}

                            
                            

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

                        isomer_xtb_exit_codes[k] = exit_code

                        if os.path.exists(isomer_path / f"conf{k}"):
                            shutil.rmtree(isomer_path / f"conf{k}")
                        shutil.copytree(conf_name, isomer_path / f"conf{k}")
                        count += 1

                    isomer.set_tag("XTB_exit_codes", json.dumps(isomer_xtb_exit_codes))
                    isomer.set_tag("XTB_geometries", json.dumps(isomer_final_geometries))
                    isomer.set_tag("xtb_energy", json.dumps(isomer_energies))
                    isomer.set_tag("xtb_free_energy", json.dumps(isomer_free_energies))
                    isomer.set_tag("isomer_free_energy_corrections", json.dumps(isomer_free_energy_corrections))
                    isomer.set_tag("xtb_lowest_conformer", json.dumps(isomer_min_energy))
                
                if self.mode.value == 'mm':
                    isomer_gfnff_exit_codes = {}
                    isomer_gfnff_geometries = {} 
                    ruggt = cast(Path, iso_dirname)
                    out_name =  str(ruggt / "xtbopt.xyz")
                    gfnff_stdout = results[count].stdout.decode()
                
                    exit_code = 1
                    pattern = 'GEOMETRY OPTIMIZATION CONVERGED AFTER'
                    for line in gfnff_stdout.split("\n"):
                            if pattern in line:
                                exit_code = 0

                    if exit_code == 0:
                        isomer_gfnff_geometries[j] = Xtb.get_trajectory(Path(out_name))
                        isomer_gfnff_exit_codes[j] = exit_code
                    else:
                        self.logger.warning("XTB GFN-FF pre-optimisation failed for '%s'", isomer.get_tag('component'))
                        isomer_gfnff_exit_codes[j] = exit_code

                    count += 1

                    
                    isomer_path = mol_path / f"{isomer.get_tag('component')}"
                    if os.path.exists(isomer_path):
                        shutil.rmtree(isomer_path)
                    shutil.copytree(cast(Path, iso_dirname), isomer_path)

                    isomer.set_tag('gfnff_geometries', json.dumps(isomer_gfnff_geometries))
                    isomer.set_tag('gfnff_exit_codes', json.dumps(isomer_gfnff_exit_codes))
                    


    def run(self) -> None:
        mode_calc = str(self.mode.value)
        
        mols = self.inp.receive()
    
        commands: list[str] = []
        res_paths: list[Path] = []
        mol_outputs: list[list[list[Path] | Path]] = []

        for i, mol in enumerate(mols):
            mol_path = Path(f"mol-{i}").absolute()
            mol_path.mkdir()
            isomer_outputs: list[list[Path] | Path] = []
            self.logger.info(f"XTB optimisation for molecule {i}: {mol.molecules[0].get_tag('component')}")

            for j, isomer in enumerate(mol.molecules):
                self.logger.info(f"{mol.molecules[0].get_tag('component')}: XTB optimisation for isomer {j}: {isomer}")
                isomer.set_tag("xtb_iso_idx", f"{i}_{j}")
                
                

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

                if mode_calc == 'sqm':
                
                    conformer_outputs: list[Path] = []
                    conformer_tag_dict = {}

                    for k, conformer in enumerate(isomer.conformers):
                        self.logger.info(f"{mol.molecules[0].get_tag('component')} - isomer {j}: XTB optimisation for conformer {k}")
                        conformer_tag_dict[k] = f"{i}_{j}_{k}"
                        conf_path = iso_path / f"conformer-{k}"
                        conf_path.mkdir()
                        res_paths.append(conf_path)
                        input_flname = f"{isomer.get_tag('component')}_conf{k}_inp.xyz"
                        input_path = conf_path / input_flname
                        output_dirname = conf_path

                        
                        keywords = (
                            f"-P {self.n_threads_job.value} "
                            f"-c {str(int(isomer_charge))} -u {str(int(isomer_mult - 1))} "
                            f"--alpb {self.solvent.value}")

                        

                        try:
                            conformer.to_xyz(path=input_path, tag_name=f"{i}_{j}_{k}")
                        except ValueError as err:
                            self.logger.warning(
                                "Skipping '%s' due to XYZ conversion error:\n %s", conformer, err
                            )

                        if self.opt.value:
                            keywords += " --ohess vtight"
                            controls = f"--input {create_xcontrol_xtb(isomer, conf_path)}"
                            keywords += f" {controls}"

                        command = f"{self.runnable['xtb']} {input_path} {keywords}"
                        commands.append(command)
                        conformer_outputs.append(output_dirname)
                    isomer_outputs.append(conformer_outputs)
                
                elif mode_calc == 'mm':
                    input_flname = f"{isomer.get_tag('component')}_MM_inp"
                    sdf_flname = input_flname + ".sdf"
                    xyz_flname = input_flname + ".xyz"
                    input_path_sdf = iso_path / sdf_flname
                    input_path_xyz = iso_path / xyz_flname
                    keywords =  f"--opt -c {str(int(isomer_charge))} -u {str(int(isomer_mult - 1))} "
                    output_dirname = iso_path
                    res_paths.append(iso_path)

                    isomer.to_sdf(Path(input_path_sdf))
                    
                    sdf_mol = Loader.molecule(Loader(str(input_path_sdf)))
                    sdf_mol.write_xyz(Path(input_path_xyz))

                    if isomer.has_tag("constraints"):
                                                
                        self.logger.info(f"found constraint for isomer {j}. Geometry Constraints and Symmetry controls added to xcontrol file.")
                        controls = f"--input {create_xcontrol_xtb(isomer, iso_path, True)}"
                        keywords += f" {controls}"

                    else:
                        self.logger.info(f"No constraints for isomer {j}. Only symmetry controls added to xcontrol file.")
                        controls = f"--input {create_xcontrol_xtb(isomer, iso_path)}"
                        keywords += f" {controls}"

                    command = f"{self.runnable['xtb']} --gfnff {input_path_xyz} {keywords}"
                    commands.append(command)
                    
                    isomer_outputs.append(output_dirname)


            mol_outputs.append(isomer_outputs)

        self.logger.info(f"Commands before run_multi: {commands}")

        # Run all commands at once
        results = self.run_multi(
            commands,
            working_dirs=res_paths,
            verbose=False,
            raise_on_failure=False,
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
