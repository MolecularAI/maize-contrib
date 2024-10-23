""""Schrodinger Prepwizard protein preparation interface"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Annotated

import pytest

from maize.core.interface import Input, Output, Parameter, Suffix
from maize.utilities.testing import TestRig
from maize.utilities.validation import SuccessValidator, FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger, has_license
from maize.utilities.io import Config


class Prepwizard(Schrodinger):
    """
    Calls Schrodinger's prepwizard util to prepare HOLO structures for docking.
    <https://www.schrodinger.com/sites/default/files/s3/release/current/Documentation/html/utilities/program_utility_usage/prepwizard.html>`_ for details.
    """
    tags = {"chemistry", "docking", "preparation"}

    required_callables = ["prepwizard"]

    inp: Input[Annotated[Path, Suffix("pdb")]] = Input()
    """Protein to prepare, must be holo PBD"""

    out: Output[Annotated[Path, Suffix("mae")]] = Output()
    """Path to prepared protein file"""

    fill_loops: Parameter[bool] = Parameter(default=False)
    """ use prime to fill missing loops """

    fill_side_chains: Parameter[bool] = Parameter(default=False)
    """ use prime to fill missing side chains """

    def run(self) -> None:
        self.logger.info("starting prepwizard...")
        input_file = self.inp.receive()

        output = Path(f"{input_file.stem}_prepared.mae")

        command_extra = " -WAIT "

        if self.fill_loops.value:
            command_extra += "-fillloops "
        if self.fill_side_chains.value:
            command_extra += "-fillsidechains "
        command = (
            f"{self.runnable['prepwizard']}"
            + command_extra
            + f"{input_file.as_posix()} {output.as_posix()}"
        )
        self.run_command(command, validators=[SuccessValidator("JobId:"), FileValidator(output)])
        self.out.send(output)


@pytest.mark.skipif(not has_license(), reason="No Schrodinger license found")
class TestSuitePrepwizard:
    @pytest.mark.needs_node("prepwizard")
    def test_prepwizard(
        self, temp_working_dir: Path, test_config: Config, input_protein: Path
    ) -> None:
        rig = TestRig(Prepwizard, config=test_config)
        res = rig.setup_run(inputs={"inp": [input_protein]})
        proc_path = res["out"].get()
        assert proc_path is not None
        assert proc_path.exists()
