from pathlib import Path
from typing import Annotated, Literal, cast

import pytest

from rdkit import Chem

from maize.core.interface import Input, Output, Parameter, Suffix
from maize.utilities.testing import TestRig
from maize.utilities.validation import FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger
from maize.utilities.io import Config

from maize.utilities.chem import (
    IsomerCollection,
    Isomer,
    save_sdf_library,
    load_sdf_or_mae_library,
    merge_libraries,
)


class SchrodingerTugAlignement(Schrodinger):
    """
    Calls Schrödinger's align_ligands util to align structures.
    Expects $SCHRODINGER/run as executable
    """

    inp: Input[list[IsomerCollection]] = Input()
    """Maize IsomerCollection objects as input"""

    align_reference: Input[Isomer] = Input()
    """align_reference Maize Isomer"""

    out: Output[list[IsomerCollection]] = Output()
    """ output of align"""

    required_callables = Schrodinger.required_callables + ["schrodinger_run"]  # should be just $SCHRODINGER/run

    def run(self) -> None:
        self.logger.info("starting align_ligands...")

        # load the inputs
        input_mols = self.inp.receive()
        # tag the input isomers
        for i, mol in enumerate(input_mols):
            for j, iso in enumerate(mol.molecules):
                iso.set_tag("m_molid", str(i))
                iso.set_tag("m_isoid", str(j))

        ref_mol = self.align_reference.receive()

        # set paths
        input_path = Path("input.sdf")
        ref_path = Path("ref.sdf")
        output_path = Path("input_aligned2_ref.maegz")  # this one comes from schrödinger itself

        save_sdf_library(input_path, input_mols, split_strategy="schrodinger")
        ref_mol.to_sdf(ref_path)

        validators = [FileValidator(output_path)]
        self.logger.info(
            f"generating output {output_path} from {input_path} by aligning to {ref_path}"
        )

        command = (
            f"{self.runnable['schrodinger_run']} "
            + f"-FROM psp tug_align.py --maintain_order --mcs_method rdkit "
            + f"{input_path.as_posix()} "
            + f"{ref_path.as_posix()}"
        )

        self.run_command(command, validators=validators, raise_on_failure=False, verbose=True)

        # Create an empty placeholder list of mols
        mols = input_mols

        # parse output
        self.logger.info("Parsing output")
        if output_path.exists():
            aligned = load_sdf_or_mae_library(output_path, split_strategy="schrodinger", renumber=False)
            self.logger.debug("Received %s aligned molecules", len(aligned))
            # Overwrite with successfully generated mols
            for mol in aligned:
                original_index = int(cast(int, mol.molecules[0].get_tag("i_sd_m\\_molid")))
                # the ugly linebreak here is introduced by Schrödinger because tug-align can only output .maegz
                mols[original_index] = mol
        else:
            self.logger.warn("No aligned output received")

        self.out.send(mols)


@pytest.fixture
def test_reference_input(shared_datadir: Path) -> Path:
    return shared_datadir / "conformer_1.sdf"
@pytest.fixture
def test_query_input(shared_datadir: Path) -> Path:
    return shared_datadir / "conformer_9.sdf"

class TestSuiteSchrodingerConverter:
    @pytest.mark.needs_node("schrodingertugalignement")
    def test_SchrodingerTugAlignement(
        self, temp_working_dir: Path, test_config: Config, test_query_input: Path, test_reference_input: Path) -> None:
        rig = TestRig(SchrodingerTugAlignement, config=test_config)

        input_struct = [IsomerCollection.from_sdf(test_query_input)]
        reference_struct = Isomer.from_sdf(test_reference_input)
        res = rig.setup_run(inputs={"inp": [input_struct], "align_reference": [reference_struct]})
        aligned =  res["out"].get()
        assert aligned is not None
        rms_score = Chem.rdMolAlign.CalcRMS(aligned[0].molecules[0]._molecule,reference_struct._molecule)
        assert rms_score <= 0.5