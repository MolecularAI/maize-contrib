""""Schrodinger glide grid generator interface"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Annotated

from maize.core.interface import Input, Output, Parameter, Suffix
from maize.utilities.validation import SuccessValidator, FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger


class GlideGridGenerator(Schrodinger):
    """
    Calls Schrodinger's generate_glide_grids util to convert prepared HOLO structures to docking grids.
    """
    tags = {"chemistry", "docking", "preparation"}

    required_callables = ["generate_glide_grids"]

    inp: Input[Annotated[Path, Suffix("mae")]] = Input()
    """Protein to convert, must be holo and prepared"""

    out: Output[Annotated[Path, Suffix("zip")]] = Output()
    """Path to prepared grid file"""

    ligand_asl: Parameter[str] = Parameter(default="'res.ptype \"INH \"'")
    """ Schrodinger asl for ligand identification"""

    def run(self) -> None:
        self.logger.info("starting grid generation...")
        input_path = self.inp.receive()

        target_output_path = Path(input_path.stem + "-gridgen.zip")

        command = (
            f"{self.runnable['generate_glide_grids']} -HOST {self.host.value} -WAIT "
            + f"-j {input_path.stem} "
            + f"-lig_asl {self.ligand_asl.value} "
            + f"-rec {input_path.as_posix()}"
        )
        self.run_command(
            command, validators=[SuccessValidator("JobId:"), FileValidator(target_output_path)]
        )
        self.out.send(target_output_path)
