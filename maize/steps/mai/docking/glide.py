"""Schrodinger GLIDE docking interface"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Annotated, Literal, Any

import pytest

from maize.core.interface import Input, Output, Parameter, Suffix
from maize.utilities.testing import TestRig
from maize.utilities.validation import SuccessValidator, FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger, has_license
from maize.utilities.chem import (
    IsomerCollection,
    Isomer,
    load_sdf_library,
    save_sdf_library,
    merge_libraries,
)


GlideConfigType = dict[str, str | int | float | bool | Path]


def _write_glide_input(path: Path, data: GlideConfigType) -> None:
    """Writes a GLIDE ``.in`` input file"""
    with path.open("w") as file:
        for key, value in data.items():
            match value:
                case Path():
                    file.write(f"{key.upper()}  {value.as_posix()}\n")
                case _:
                    file.write(f"{key.upper()}  {value}\n")


class Glide(Schrodinger):
    """
    Calls Schrodinger's GLIDE to dock small molecules.

    Notes
    -----
    Due to Schrodinger's licensing system, each call to a tool requires
    going through Schrodinger's job server. This is run separately for
    each job to avoid conflicts with a potentially running main server.

    See Also
    --------
    :class:`~maize.steps.mai.docking.Vina` :
        A popular open-source docking program
    :class:`~maize.steps.mai.docking.AutoDockGPU` :
        Another popular open-source docking tool with GPU support

    """

    N_LICENSES = 4
    DEFAULT_OUTPUT_NAME = "glide"
    GLIDE_SCORE_TAG = "r_i_docking_score"

    required_callables = ["glide"]

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to dock"""

    inp_grid: Input[Annotated[Path, Suffix("zip")]] = Input()
    """Previously prepared GLIDE grid file"""

    ref_ligand: Input[Isomer] = Input(optional=True)
    """Optional reference ligand"""

    out: Output[list[IsomerCollection]] = Output()
    """Docked molecules with poses and energies included"""

    precision: Parameter[Literal["SP", "XP", "HTVS"]] = Parameter(default="SP")
    """GLIDE docking precision"""

    keywords: Parameter[GlideConfigType] = Parameter(default_factory=dict)
    """
    Additional GLIDE keywords to use, see the `GLIDE documentation
    <https://www.schrodinger.com/sites/default/files/s3/release/current/Documentation/html/glide/glide_command_reference/glide_command_glide.htm>`_ for details.

    """

    def run(self) -> None:
        mols = self.inp.receive()
        inp_file = Path("input.sdf")
        grid_obj = self.inp_grid.receive()

        config: GlideConfigType = {
            "GRIDFILE": grid_obj.as_posix(),
            "PRECISION": self.precision.value,
            "LIGANDFILE": inp_file,
            "POSE_OUTTYPE": "ligandlib_sd",
            "POSES_PER_LIG": 4,
            "COMPRESS_POSES": False,
            "NOSORT": True,
        }
        config.update(self.keywords.value)

        # Optional reference ligand
        if self.ref_ligand.is_set:
            ref_path = Path("ref.sdf")
            ref = self.ref_ligand.receive()
            ref.to_sdf(ref_path)
            self.logger.info("Using reference ligand '%s'", ref.to_smiles())
            config["REF_LIGAND_FILE"] = ref_path
            config["USE_REF_LIGAND"] = True
            config["CORE_RESTRAIN"] = True
            config["CORE_DEFINITION"] = "mcssmarts"

        save_sdf_library(inp_file, mols, split_strategy="schrodinger")
        glide_inp_file = Path("glide.in")
        _write_glide_input(glide_inp_file, config)
        self.logger.debug("Prepared GLIDE input for %s molecules", len(mols))

        # Wait for licenses
        self.logger.info("Waiting for %s licenses...", self.N_LICENSES * self.n_jobs.value)
        key = "GLIDE_XP_DOCKING" if self.precision.value == "XP" else "GLIDE_SP_DOCKING"
        self.guard.wait(key, number=self.N_LICENSES * self.n_jobs.value)

        # Run
        self.logger.info("Found licenses, docking...")
        output = Path(f"{self.DEFAULT_OUTPUT_NAME}_raw.sdf")
        command = (
            f"{self.runnable['glide']} -HOST {self.host.value} -WAIT "
            f"-NJOBS {self.n_jobs.value} -JOBNAME {self.DEFAULT_OUTPUT_NAME} "
            f"{glide_inp_file.as_posix()}"
        )
        self.run_command(command, validators=[SuccessValidator("JobId:"), FileValidator(output)])

        self.logger.info("Parsing output")
        docked = load_sdf_library(output, split_strategy="schrodinger")
        for mol in docked:
            for iso in mol.molecules:
                iso.score_tag = self.GLIDE_SCORE_TAG
        mols = merge_libraries(mols, docked)

        self.out.send(mols)


# From IcolosData
@pytest.fixture
def grid(shared_datadir: Any) -> Any:
    return shared_datadir / "1UYD_grid_no_constraints.zip"


@pytest.mark.skipif(not has_license(), reason="No Schrodinger license found")
class TestSuiteGlide:
    def test_Glide(
        self, temp_working_dir: Any, test_config: Any, example_smiles: Any, grid: Any
    ) -> None:
        rig = TestRig(Glide, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in example_smiles]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs]}, parameters={"inp_grid": grid})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 5
        assert mols[0].molecules[0].n_conformers == 4
        assert -8 < mols[0].molecules[0].scores[0] < -5
        assert mols[0].n_isomers == 1
