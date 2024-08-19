""""Schrodinger structconvert interface"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Annotated, Literal

import pytest

from maize.core.interface import Input, Output, Parameter, Suffix
from maize.utilities.testing import TestRig
from maize.utilities.validation import FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger
from maize.utilities.io import Config


class SchrodingerConverter(Schrodinger):
    """
    Calls Schrodinger's structconvert util to convert between formats.
    Mainly useful to hanlde  proprietary mae/gz formats.
    Expects $SCHRODINGER/utilities/structconvert as executable
    """

    inp: Input[Annotated[Path, Suffix("mae", "sdf", "pdb")]] = Input()
    """ input to converter, mae, sd or pdb"""

    out: Output[Annotated[Path, Suffix("mae", "sdf", "pdb")]] = Output()
    """ output of converter"""

    output_type: Parameter[Literal[".mae", ".sdf", ".pdb"]] = Parameter(optional=True)
    """
    Target output type of converter, should be different to input.
    Inferred by default .mae->.sdf, .sdf->.mae, .pdb->.mae

    """

    required_callables = ["structconvert"]  # normally $SCHRODINGER/utilities/structconvert

    def run(self) -> None:
        self.logger.info("starting structconvert...")
        input_path = self.inp.receive()
        input_suffix = input_path.suffix
        if self.output_type.is_set:
            output_suffix = self.output_type.value
        elif input_suffix == ".mae":
            output_suffix = ".sdf"
        elif input_suffix == ".sdf":
            output_suffix = ".mae"
        elif input_suffix == ".pdb":
            output_suffix = ".mae"
        else:
            raise ValueError(f"incompatible input file type: {input_suffix}, must be .sdf or .mae")

        output_path = input_path.with_suffix(output_suffix)

        validators = [FileValidator(output_path)]
        self.logger.info(f"generating output {output_path} from {input_path}")

        command = (
            f"{self.runnable['structconvert']} "
            + f"{input_path.as_posix()} "
            + f"{output_path.as_posix()}"
        )

        self.run_command(command, validators=validators, raise_on_failure=False, verbose=True)
        self.out.send(output_path)


@pytest.fixture
def pdb_input(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_apo.pdb"


@pytest.fixture
def mae_input(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_apo.mae"


class TestSuiteSchrodingerConverter:
    @pytest.mark.needs_node("schrodingerconverter")
    def test_SchrodingerConverter(
        self, temp_working_dir: Path, test_config: Config, pdb_input: Path, mae_input: Path
    ) -> None:
        rig = TestRig(SchrodingerConverter, config=test_config)

        res = rig.setup_run(inputs={"inp": [pdb_input]})
        output_file = res["out"].get()
        assert output_file is not None

        res = rig.setup_run(inputs={"inp": [pdb_input]}, parameters={"output_type": ".mae"})
        output_file = res["out"].get()
        assert output_file is not None

        res = rig.setup_run(inputs={"inp": [mae_input]}, parameters={"output_type": ".pdb"})
        output_file = res["out"].get()
        assert output_file is not None
