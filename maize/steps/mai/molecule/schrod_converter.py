""""Schrodinger sdconvert interface"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Annotated, Literal, Any, List, Union

import pytest

from maize.core.interface import Input, Output, Parameter, FileParameter, Suffix
from maize.utilities.testing import TestRig
from maize.utilities.validation import SuccessValidator, FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger



class SchrodingerConverter(Schrodinger, register=False):
    """
    Calls Schrodinger's sdconvert util to convert between formats.
    Mainly useful to  proprietary mae/gz formats
    Base class for more conveniently packaged nodes
   """

    inp:  Input[Annotated[Path, Suffix("mae","sdf")]] = Input()
    """ input to converter, mae or sd"""

    out: Output[Annotated[Path, Suffix("mae","sdf")]] = Output()
    """ type of split, must be set by derived classes"""

    required_callables = ["sdconvert"] # normally $SCHRODINGER/utilities/sdconvert

    def run(self) -> None:

        self.logger.info("starting sdconvert...")
        input_path = self.inp.receive()
        input_suffix = input_path.suffix
        if input_suffix == ".mae":
            output_path = input_path.with_suffix(".sdf")
            input_format = "imae"
            output_format = "osd"
        elif input_suffix == ".sdf":
            output_path = input_path.with_suffix(".mae")
            input_format = "isd"
            output_format = "omae"
        else:
            ValueError(f"incompatible input file type: {input_suffix}, must be .sdf or .mae")

        validators = [FileValidator(output_path)]
        self.logger.info(f"generating output {output_path} from {input_path}")


        command = (
                f"{self.runnable['sdconvert']} " +
                f"-{input_format} {input_path.as_posix()} "+
                f"-{output_format} {output_path.as_posix()}"
        )

        self.run_command(command,
                         validators=validators,
                         raise_on_failure=False, verbose=True)
        self.out.send(output_path)



