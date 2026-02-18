"""Peptide solubility prediction using CamSol"""

import logging
from pathlib import Path
import numpy as np
import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output
from maize.utilities.chem import IsomerCollection
from maize.utilities.validation import FileValidator
from maize.utilities.io import Config
from maize.utilities.testing import TestRig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SEPARATOR_TOKEN = "|"


class Camsol(Node):
    """
    Run CAMSOL model for predicting the solubility of peptides.
    """

    required_callables = ["camsol"]
    inp: Input[list[str]] = Input()
    out: Output[list[IsomerCollection]] = Output()

    def run(self) -> None:
        input_smiles = self.inp.receive()

        # Check length of input sequence, if less than 7 amino acids, set score to be None
        valid_inputs = []
        valid_inputs_index = []
        for i, smi in enumerate(input_smiles):
            if smi.count(SEPARATOR_TOKEN) < 6:
                logging.warning(
                    f"The input sequence {smi} contains less than 7 amino acids. "
                    "Set score to be NaN!"
                )
            else:
                valid_inputs.append(smi)
                valid_inputs_index.append(i)

        valid_inputs_smi = self._prepare_input_sequence(valid_inputs)
        input_path = self._prepare_input_file(valid_inputs_smi)

        outfilename_to_camsol = "camsol_output"  # camsol internally adds .txt extension
        camsol_output_file_path = Path(f"{outfilename_to_camsol}.txt")
        # set validator
        validators = [FileValidator(camsol_output_file_path)]

        # command to run CAMSOL when using conda env
        command = (
            f"{self.runnable['camsol']} "
            f"{input_path} "
            f"--outfilename {outfilename_to_camsol} "
        )

        self.run_command(command, verbose=True, validators=validators)

        scores = [float("nan")] * len(input_smiles)
        valid_scores = self._parse_output(camsol_output_file_path)
        if len(valid_inputs_index) == len(valid_scores):
            for i, index in enumerate(valid_inputs_index):
                scores[index] = valid_scores[i]
        else:
            raise ValueError(
                f"Number of valid inputs {len(valid_inputs_index)} "
                f"does not match number of scores {len(valid_scores)}"
            )

        mols = []
        for smi, score in zip(input_smiles, scores):
            # note that clean_smi_for_mol is only for creating IsomerCollection without error
            # and it does not correspond to the original peptide smiles from pepinvent
            # because the cyclization numbers are removed and O is added to the end of each
            # amino acid. The mol in IsomerCollection is not used anyway in reinvent and only
            # the scores are used.
            clean_smi_for_mol = smi.replace("|", "").replace("||", "")
            mol = IsomerCollection.from_smiles(clean_smi_for_mol, max_isomers=1)
            mol.smiles = smi
            mol.add_score("camsol_score", score)
            mols.append(mol)

        self.out.send(mols)

    def _prepare_input_sequence(self, peptide_fragmented_amino_acids: list[str]) -> list[str]:
        """Prepare the input sequence for CAMSOL.
        @param peptide_fragmented_amino_acids: list of fragmented amino acids, e.g 
            ["N[C@@H](CC(=O)O)C(=O)O|N[C@@H](CS)C(=O)O|N[C@@H](CO)C(=O)O"], 
            which requires:
                - each amino acid ended with C(=O)O
                - no cyclization numbers in each amino acid
        @return: list of input sequence for CAMSOL with each sequence having following requirements
                - start and end with |
                - each amino acid separated by ||
                - each amino acid ended with C(=O)O
                - no cyclization numbers in each amino acid
                e.g. ["|N[C@@H](CC(=O)O)C(=O)O||N[C@@H](CS)C(=O)O||N[C@@H](CO)C(=O)O|"]
        """
        prepared_sequences = []
        for smi in peptide_fragmented_amino_acids:
            prepared_sequence = "|" + smi.replace("|", "||") + "|"
            prepared_sequences.append(prepared_sequence)
        return prepared_sequences

    def _prepare_input_file(self, input_smiles: list[str]) -> Path:
        """Prepare the input file for CAMSOL."""

        input_path = Path("input_fasta.txt")
        logging.info(f"Preparing input file for CAMSOL to {input_path.absolute()}")

        # Write the input list of SMILES to the file in fasta format
        with open(input_path, "w") as f:
            for i, seq in enumerate(input_smiles):
                f.write(f">Input_{i+1}\n")
                f.write(f"{seq}\n")

        return input_path

    def _parse_output(self, camsol_output_file_path: Path) -> list[float]:
        """
        Parse the output file and return the camsol scores as a list of floats.

        # The output file looks like this
        # Name	pH	protein variant score	intrinsic solubility profile
        # Input_1	7.00	-1.468736	-1.753990045729269;-1.8978656832893255;-2.049374260300931;...
        # Input_2	7.00	-1.642780	-1.7475618171715708;-1.883200481732586;-1.995028272135342;...

        """

        logging.info(f"Parsing output file {camsol_output_file_path.absolute()}")
        with open(camsol_output_file_path, "r") as f:
            lines = f.readlines()
        scores = []
        for line in lines:
            if line.startswith("Name"):
                continue
            columns = [field.strip() for field in line.strip().split("\t")]
            if len(columns) >= 3:
                try:
                    scores.append(float(columns[2]))
                except ValueError:
                    logging.error(f"Parse error: {line}")

        return scores

@pytest.mark.needs_node("camsol")
def test_camsol(test_config: Config) -> None:
    smiles = [
        "N[C@@H](CC(=O)O)C(=O)O|N[C@@H](CS)C(=O)O|NCC(=O)O|N[C@@H](CCSC)C(=O)O|"
            "N1[C@@H](CCC1)C(=O)O|N[C@@H](C(C)C)C(=O)O|N[C@@H](CO)C(=O)O",
        "N[C@@H](CC(=O)O)C(=O)O|N[C@@H](CS)C(=O)O|NCC(=O)O|N[C@@H](CCSC)C(=O)O|N1[C@@H](CCC1)C(=O)O"
          ]
    rig = TestRig(Camsol, config=test_config)
    res = rig.setup_run(inputs={"inp": [smiles]})
    data = res["out"].get()

    assert data is not None
    assert hasattr(data, "__len__")
    assert len(data) == len(smiles)
    scores = []
    expected_scores = [0.3801, float("nan")]
    for mol in data:
        scores.append(round(mol.scores["camsol_score"], 4))
    np.testing.assert_array_equal(scores, expected_scores)
