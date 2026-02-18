"""Boltz-2 for affinity prediction"""

# pylint: disable=import-outside-toplevel, import-error

from dataclasses import dataclass, field
import itertools
from pathlib import Path
from typing import Annotated, List, cast

import json
import numpy as np
import pytest
import yaml

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Suffix, FileParameter
from maize.utilities.testing import TestRig
from maize.utilities.chem import  ChemistryException, IsomerCollection, Isomer
from maize.utilities.io import Config

SCORE_TAGS = [
    "ligand_plddt",
    "iptm",
    "confidence_score",
    "complex_iplddt",
    "ligand_iptm",
    "affinity_pred_value",
    "affinity_probability_binary",
]
SCORE_AGG = {
    "affinity_pred_value": "min",
    # All others use 'max'
}


class Boltz2(Node):
    """
    Calls Boltz-2 as a docking tool, returning the bound poses and the predicted affinities from an input
    isomer collection. Because Boltz-2 does not do enumeration, it is better ot

    """

    tags = {"chemistry", "docking", "scorer", "tagger"}

    required_callables = ["boltz"]

    inp: Input[list[str]] = Input()
    """"Smiles strings to dock"""

    inp_seq: Parameter[str] = Parameter()
    """FASTA sequence for the target"""

    inp_msa: FileParameter[Annotated[Path, Suffix("a3m")]] = FileParameter()
    """MSA file precomputed for the target, optinla but reccomended"""

    protein_template: FileParameter[Annotated[Path, Suffix("cif")]] = FileParameter(optional=True)
    """A CIF file with known structure for the target, optional"""

    pocket_residue_constraints: Parameter[List[int]] = Parameter(optional=True)
    """Pocket residue numbers that should be close to the binder"""

    pocket_constraint_max_distance: Parameter[float] = Parameter(default=2.0)
    """Max distance to the inhibitor from the pocket constraints"""

    recycling_steps: Parameter[int] = Parameter(default=3)
    """Number of recycling steps for the Boltz-2 prediction."""

    sampling_steps: Parameter[int] = Parameter(default=50)
    """Number of sampling steps for the diffusion process."""

    diffusion_samples: Parameter[int] = Parameter(default=1)
    """Number of diffusion samples to generate."""

    diffusion_samples_affinity: Parameter[int] = Parameter(default=1)
    """Number of diffusion samples to use for affinity."""

    sampling_steps_affinity: Parameter[int] = Parameter(default=200)
    """Number of sampling steps for affinity prediction."""

    out: Output[list[IsomerCollection]] = Output()
    """Docked molecules"""

    use_potentials: Parameter[bool] = Parameter(default=False)
    """Whether to use physics-inspired potentials"""

    num_workers: Parameter[int] = Parameter(default=2)
    """Number of workers for Boltz-2 prediction"""

    max_parallel_samples: Parameter[int] = Parameter(default=5)
    """Maximum number of samples to process in parallel"""

    no_kernels: Parameter[bool] = Parameter(default=True)
    """Whether to disable custom Nvidia kernels. Need a compatible GPU to use kernels."""

    def run(self) -> None:

        # ensure smiles in yamls are in quotation marks
        def quoted_presenter(dumper, data) -> yaml.ScalarNode:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="''")

        yaml.add_representer(str, quoted_presenter)

        smiles = self.inp.receive()

        # make a folder to hold input files

        # set up the folder hierarchy
        # Boltz wants all inputs stored in one folder
        input_dir = Path("boltz-input-dir")
        input_dir.mkdir(exist_ok=True)
        # and takes an output path to store the results
        output_dir = Path("boltz-output-dir")
        output_dir.mkdir(exist_ok=True)
        # the expected path to affinity data is then
        path_to_look_for_data = output_dir / f"boltz_results_{input_dir.name}" / "predictions"

        # handle inputs - create an isomer collection
        mols = []
        for mol_index, smi in enumerate(smiles):
            # create an iso collection for each input
            ic = IsomerCollection.from_smiles(smi, timeout=60, max_isomers=2) # limit to 2 isomers for speed,
            # not clear how much Boltz cares about stereo
            # If isomer collection is empty, skip the rest of the loop
            mols.append(ic)

            if ic.n_isomers == 0:
                self.logger.warning(f"No isomers generated for SMILES '{smi}' (index {mol_index}), skipping.")
                continue
            identifier = f"m_molid_{mol_index}"
            input_path = input_dir / (identifier + ".yaml")
            config = {
                "sequences": [
                    {
                        "protein": {
                            "id": ["A"],
                            "sequence": self.inp_seq.value,
                            "msa": self.inp_msa.value.as_posix(),
                        }
                    },
                    {
                        "ligand": {
                            "id": ["I"],
                            "smiles": ic.molecules[0].to_smiles(remove_h=True),
                        }
                    },
                ],
                "properties": [{"affinity": {"binder": "I"}}],
            }

            if self.protein_template.is_set:
                config["templates"] = [{"cif": self.protein_template.value}]
            if self.pocket_residue_constraints.is_set:
                config["constraints"] = [
                    {
                        "pocket": {
                            "binder": "I",
                            "contacts": [],
                            "max_distance": self.pocket_constraint_max_distance.value,
                        }
                    }
                ]
                for residue_index in cast(List[int], self.pocket_residue_constraints.value):
                    config["constraints"][0]["pocket"]["contacts"].append(["A", residue_index])

            ## write the config to a yaml file
            with open(input_path, "w") as f:
                yaml.dump(config, f)
            self.logger.debug(f"dumping Boltz2 input yaml to {input_path} ")

        ## create the command to run Boltz-2
        cmd = (
            f"{self.runnable['boltz']} predict "
            f"--out_dir {output_dir.as_posix()} "
            f"--recycling_steps {self.recycling_steps.value} "
            f"--sampling_steps {self.sampling_steps.value} "
            f"--diffusion_samples {self.diffusion_samples.value} "
            f"--max_parallel_samples {self.max_parallel_samples.value} "
            f"--output_format pdb "
            + ("--use_potentials " if self.use_potentials.value else "")
            + f"--sampling_steps_affinity {self.sampling_steps_affinity.value} "
            f"--diffusion_samples_affinity {self.diffusion_samples_affinity.value} "
            f"--step_scale '1.638' "
            f"--num_workers {self.num_workers.value} "
            + ("--no_kernels " if self.no_kernels.value else "")
            + f"--override "
            f" {input_dir.as_posix()} "
        )

        self.logger.info(f"running commmand {cmd} ")

        # run the command
        self.run_command(cmd)

        # collect all predictions and uncertainties

        results_dictionary = dict()
        for subdir in path_to_look_for_data.iterdir():
            self.logger.debug(
                f"Scanning predictions directory: {subdir.name} for molecule results."
            )

            if subdir.is_dir():
                result_name = subdir.name
                mol_index = int(result_name.split("_")[2])

                # collect affinity - we will use the ensemble affinity prediction for all poses
                expect_res = subdir / f"affinity_{result_name}.json"
                self.logger.debug(
                    f"Checking for affinity results file: {expect_res} for molecule index {mol_index}."
                )

                if expect_res.is_file():
                    self.logger.info(
                        f"Affinity predictions found for molecule '{result_name}' (index {mol_index})."
                    )
                    with expect_res.open("r") as f:
                        affinity_data = json.load(f)
                        # note that this will store only the aggregate affinity per molecule, even if multiple poses, as the affinity samples are unconnected from the structure prediction poses
                # now we need to loop over diffusion samples/poses to collect pose-specific data
                for iso_index in range(self.diffusion_samples.value):
                    key = (mol_index, iso_index)  # index to store results

                    # save the affinity values
                    results_dictionary[key] = {
                        "affinity_pred_value": affinity_data["affinity_pred_value"],
                        "affinity_probability_binary": affinity_data["affinity_probability_binary"],
                    }

                    # collect confidence metrics per pose
                    expect_res = subdir / f"confidence_{result_name}_model_{iso_index}.json"
                    self.logger.debug(
                        f"Checking for confidence metrics file: {expect_res} for molecule index {mol_index}, pose {iso_index}."
                    )

                    if expect_res.is_file():
                        self.logger.info(
                            f"Confidence metrics found for molecule '{result_name}' (index {mol_index}), pose {iso_index}."
                        )
                        with expect_res.open("r") as f:
                            confidence_data = json.load(f)
                            # If the key already exists, update its dictionary; else, create it
                            if key in results_dictionary:
                                results_dictionary[key].update(
                                    {
                                        "confidence_score": confidence_data["confidence_score"],
                                        "iptm": confidence_data["iptm"],
                                        "complex_plddt": confidence_data["complex_plddt"],
                                        "ligand_iptm": confidence_data["ligand_iptm"],
                                        "complex_iplddt": confidence_data["complex_iplddt"],
                                    }
                                )
                            else:
                                results_dictionary[key] = {
                                    "confidence_score": confidence_data["confidence_score"],
                                    "iptm": confidence_data["iptm"],
                                    "complex_plddt": confidence_data["complex_plddt"],
                                    "ligand_iptm": confidence_data["ligand_iptm"],
                                    "complex_iplddt": confidence_data["complex_iplddt"],
                                }

                    # collect ligand structure & ligand plddt (b-factors)
                    expect_res = subdir / f"{result_name}_model_{iso_index}.pdb"
                    self.logger.info(f" looking for {expect_res}")

                    if expect_res.is_file():
                        self.logger.info(
                            f"found Boltz-2 predicted structure for {result_name}, model {iso_index}"
                        )
                        bfactors = []
                        pdb_lines = []
                        with expect_res.open("r") as f:
                            for line in f:
                                if line.startswith("HETATM"):
                                    residue_name = line[17:20].strip()
                                    if residue_name == "LIG":
                                        pdb_lines.append(line)
                                        bfactors.append(float(line[60:66]))
                        ## add pose reconstruction from PBDLines
                        ## the issue here is that since we don't get connectivity
                        ## from boltz-2, rdkit can perceive different chemical graphs
                        ## from the pdbs, which results in isos with different numbers of atoms
                        try:
                            iso = Isomer.from_pdbblock("".join(pdb_lines))
                        except ChemistryException as e:
                            self.logger.warning(f"Failed to parse PDB block for {result_name}, model {iso_index}: {e}")
                            continue  # skip storing this iso, continue with next
                        if key in results_dictionary:
                            results_dictionary[key].update(
                                {"ligand_plddt": np.mean(bfactors), "pose": iso}
                            )
                        else:
                            results_dictionary[key] = {
                                "ligand_plddt": np.mean(bfactors),
                                "pose": iso,
                            }

        #ggself.logger.debug(f"Summary Boltz results keys: {list(results_dictionary.keys())}")
        # match outputs with input molecules
        for mol_index, _ in enumerate(smiles):
            iso_list = []  # list of found poses
            for iso_index in range(self.diffusion_samples.value):

                if (mol_index, iso_index) in results_dictionary.keys():
                    iso_results = results_dictionary[(mol_index, iso_index)]
                    self.logger.debug(
                        f"found result for index {(mol_index, iso_index)}: {iso_results}"
                    )

                    if "pose" in iso_results:  # if we found a correct pose for this mol_index
                        iso = iso_results.pop("pose")
                        for key_name in iso_results.keys():
                            agg = SCORE_AGG.get(key_name, "max")
                            iso.add_score(name=key_name, value=iso_results[key_name], agg=agg)
                        iso_list.append(iso)
            if len(iso_list) > 0:
                self.logger.info(
                    f"updating index {mol_index} with {len(iso_list)} isomers from Boltz-2 results."
                )
                # create an isomer collection and overwrite the element in mols
                # in case of no results found the output will be just be an empty IC based on the input smiles
                mols[mol_index] = IsomerCollection(iso_list)
            else:
                # Add score tags with NaN values and correct aggregation
                for tag in SCORE_TAGS:
                    agg = SCORE_AGG.get(tag, "max")
                    mols[mol_index].add_score(name=tag, value=np.nan, agg=agg)
            # Set the primary score tag to 'affinity_probability_binary'
            mols[mol_index].primary_score_tag = "affinity_probability_binary"

        self.out.send(mols)


@pytest.fixture
def input_fasta() -> str:
    # Example FASTA string,  corresponding to PDB ID 4PX6 (SYK)
    return "YLDRKLLTLEDKELGSGNFGTVKKGYYQMKKVVKTVAVKILKNEANDPALKDELLAEANVMQQLDNPYIVRMIGICEAESWMLVMEMAELGPLNKYLQQNRHVTDKNIIELVHQVSMGMKYLEESNFVHRDLAARNVLLVTQHYAKISDFGLSKALRADENYYKAQTHGKWPVKWYAPECINYYKFSSKSDVWSFGVLMWEAFSYGQKPYRGMKGSEVTAMLEKGERMGCPAGCPREMYDLMNLCWTYDVENRPGFAAVELRLRNYYYDVVNE"


@pytest.fixture
def input_msa(shared_datadir: Path) -> Path:
    # Place your test MSA file in tests/data/ and update the filename as needed
    return shared_datadir / "4PX6-seq.a3m"


@pytest.fixture
def example_smiles_syk() -> list[str]:
    # Local fixture for example SMILES
    return ["[nH]1ccc(c12)cccc2Nc(c(c34)c(=O)[nH]cn4)nc(c3)NC(C5N)CCCC5"]


class TestSuiteBoltz2:
    @pytest.mark.needs_node("Boltz2")
    def test_boltz2(
        self,
        temp_working_dir: Path,
        test_config: Config,
        example_smiles_syk: list[str],
        input_fasta: str,
        input_msa: Path,
    ) -> None:
        rig = TestRig(Boltz2, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [example_smiles_syk]},
            parameters={
                "inp_seq": input_fasta,
                "inp_msa": input_msa,
                "diffusion_samples": 2,
                "diffusion_samples_affinity": 2,
                "sampling_steps": 25,
            },
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == len(example_smiles_syk)
        # Assert primary_score is approximately -2.17 ± 0.1
        assert pytest.approx(mols[0].scores["affinity_pred_value"], abs=0.1) == -2.17
        # Assert scores keys
        expected_keys = [
            "affinity_pred_value",
            "affinity_probability_binary",
            "confidence_score",
            "iptm",
            "complex_plddt",
            "ligand_iptm",
            "complex_iplddt",
            "ligand_plddt",
        ]
        assert list(mols[0].scores.keys()) == expected_keys
        # Add more asserts as needed for your Boltz2 output
