"""Schrodinger GLIDE docking interface"""

# pylint: disable=import-outside-toplevel, import-error

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, cast
from typing_extensions import Self

import numpy as np
import pytest

from maize.core.interface import Input, Output, Parameter, Suffix, FileParameter, Flag
from maize.utilities.testing import TestRig
from maize.utilities.utilities import unique_id
from maize.utilities.validation import FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger
from maize.utilities.chem import (
    IsomerCollection,
    Isomer,
    load_sdf_library,
    save_sdf_library,
    merge_libraries,
)
from maize.utilities.io import Config


GlideConfigType = dict[str, str | int | float | bool | Path | list[str]]


@dataclass
class GlideConfig:
    options: GlideConfigType
    constraints: dict[str, GlideConfigType] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Initialize from an existing Glide config"""
        current_section = None
        options: GlideConfigType = {}
        constraints: dict[str, GlideConfigType] = {}
        with path.open("r") as file:
            for line in file.readlines():
                match line.split(maxsplit=1):
                    case [cons_key] if cons_key.startswith("["):
                        current_section = cons_key
                        constraints[cons_key] = {}
                    case [key, value] if current_section:
                        constraints[cons_key][key] = value.strip()
                    case [key, value] if current_section is None:
                        options[key] = value.strip()
        return cls(options=options, constraints=constraints)

    def to_file(self, path: Path) -> None:
        """Write config to file"""
        with path.open("w") as file:
            for key, value in self.options.items():
                match value:
                    case Path():
                        file.write(f"{key.upper()}   {value.as_posix()}\n")
                    case list():
                        con = ", ".join(f'"{val}"' for val in value)
                        file.write(f"{key.upper()}   {con}\n")
                    case _:
                        file.write(f"{key.upper()}   {value}\n")

            file.write("\n")
            for name, cons in self.constraints.items():
                file.write(f"{name}\n")
                for key, value in cons.items():
                    file.write(f"    {key}   {value}\n")
                file.write("\n")

    def update(self, other: Self) -> None:
        """Join with other config"""
        self.options |= other.options
        self.constraints |= other.constraints


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
    tags = {"chemistry", "docking", "scorer", "tagger"}

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

    n_poses: Parameter[int] = Parameter(default=4)
    """How many docked poses to save"""

    precision: Parameter[Literal["SP", "XP", "HTVS"]] = Parameter(default="SP")
    """GLIDE docking precision"""

    core_restrain: Flag = Flag(default=True)
    """When using a reference ligand, whether to add restraints"""

    core_definition: Parameter[Literal["mcssmarts"]] = Parameter(default="mcssmarts")
    """When using a reference ligand, the core definition to use"""

    constraints: FileParameter[Annotated[Path, Suffix("in")]] = FileParameter(optional=True)
    """A GLIDE input file containing all desired constraints"""

    raise_on_failure: Flag = Flag(default=False)
    """Whether to raise an exception if docking failed completely, or just return ``NaN``"""

    keywords: Parameter[GlideConfigType] = Parameter(default_factory=dict)
    """
    Additional GLIDE keywords to use, see the `GLIDE documentation
    <https://www.schrodinger.com/sites/default/files/s3/release/current/Documentation/html/glide/glide_command_reference/glide_command_glide.htm>`_ for details.

    """

    max_score: Parameter[float] = Parameter(default=np.inf)
    """
    Maximum score to set for a compound. By default, compounds that don't satisfy
    constraints get a score of 10000, compounds that couldn't be docked get a score
    of NaN. To limit the former to something smaller, change this parameter.

    """

    query_interval: Parameter[int] = Parameter(default=60)

    def run(self) -> None:
        mols = self.inp.receive()
        inp_file = Path("input.sdf")
        grid_obj = self.inp_grid.receive()

        config: GlideConfigType = {
            "GRIDFILE": grid_obj.as_posix(),
            "PRECISION": self.precision.value,
            "LIGANDFILE": inp_file,
            "POSE_OUTTYPE": "ligandlib_sd",
            "POSES_PER_LIG": self.n_poses.value,
            "COMPRESS_POSES": False,
            "NOSORT": True,
        }
        config.update(self.keywords.value)

        # Optional reference ligand
        ref = self.ref_ligand.receive_optional()
        if ref:
            ref_path = Path("ref.sdf")
            ref.to_sdf(ref_path)
            self.logger.info("Using reference ligand '%s'", ref.to_smiles(remove_h=True))
            config["REF_LIGAND_FILE"] = ref_path
            config["USE_REF_LIGAND"] = True
            config["CORE_RESTRAIN"] = self.core_restrain.value
            if "CORE_DEFINITION" not in self.keywords.value.keys():
                config["CORE_DEFINITION"] = self.core_definition.value

        for i, mol in enumerate(mols):
            for j, iso in enumerate(mol.molecules):
                iso.set_tag("m_molid", str(i))
                iso.set_tag("m_isoid", str(j))

        save_sdf_library(inp_file, mols, split_strategy="schrodinger")
        glide_inp_file = Path("glide.in")

        # We get the constraints from a reference input file and add them
        # to the config, as they don't fall into a simple key-value scheme
        extra_config = GlideConfig(options=config)
        if self.constraints.is_set:
            extra_config = GlideConfig.from_file(self.constraints.filepath)

            # We can't allow a reference to be passed in with the constraints
            # file without it also being passed in through maize
            if ref is None:
                for option in [
                    "REF_LIGAND_FILE",
                    "USE_REF_LIGAND",
                    "CORE_RESTRAIN",
                    "CORE_DEFINITION",
                ]:
                    self.logger.warning("Option '%s' must be used with a reference ligand", option)
                    extra_config.options.pop(option, None)
            extra_config.update(GlideConfig(options=config))

        extra_config.to_file(glide_inp_file)
        self.logger.debug("Prepared GLIDE input for %s molecules", len(mols))

        # Run
        name = f"glide-{unique_id(12)}"
        self.logger.info("Found licenses, docking...")
        output = Path(f"{name}_raw.sdf")
        command = f"{self.runnable['glide']}"
        args = f"{glide_inp_file.as_posix()}"
        res = self._run_schrodinger_job(
            command,
            args,
            name=name,
            validators=[FileValidator(output)],
            raise_on_failure=self.raise_on_failure.value,
        )
        if res.returncode != 0:
            self.logger.warning("Glide failed, returning NaNs for all compounds")
            for mol in mols:
                for iso in mol.molecules:
                    iso.set_tag("origin", self.name)
                    iso.add_score(self.GLIDE_SCORE_TAG, np.nan)

            self.out.send(mols)
            return

        self.logger.info("Parsing output")

        if output.exists():
            docked = load_sdf_library(output, split_strategy="schrodinger", renumber=False)
            self.logger.debug("Received %s docked molecules", len(docked))
            mols = merge_libraries(mols, docked, moltag="m_molid", isotag="m_isoid", merge_conformers=True)
        else:
            self.logger.warn("No valid glide output received")

        for mol in mols:
            for iso in mol.molecules:
                iso.set_tag("origin", self.name)
                if not iso.has_tag(self.GLIDE_SCORE_TAG):
                    self.logger.warning(
                        "Docking failed for %s, no '%s' tag", iso.name or iso.inchi, self.GLIDE_SCORE_TAG
                    )
                    iso.add_score(self.GLIDE_SCORE_TAG, np.nan)
                    continue

                for conf in iso.conformers:
                    if conf.has_tag(self.GLIDE_SCORE_TAG):
                        score = float(cast(float, conf.get_tag(self.GLIDE_SCORE_TAG)))
                        score = min(score, self.max_score.value)
                        conf.add_score(self.GLIDE_SCORE_TAG, score)
                iso.conformers.sort(key=lambda conf: conf.primary_score)

                # We need to explicitly cast to float here, as Glide may sometimes set an integer
                score = float(cast(float, iso.get_tag(self.GLIDE_SCORE_TAG)))

                # Optionally limit the score to make statistics on failed compounds easier
                score = min(score, self.max_score.value)
                iso.add_score(self.GLIDE_SCORE_TAG, score)
            mol.primary_score_tag = self.GLIDE_SCORE_TAG

        self.out.send(mols)


# From IcolosData
@pytest.fixture
def grid(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_grid_no_constraints.zip"


@pytest.fixture
def grid_constraints(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_grid_constraints.zip"


@pytest.fixture
def constraint_input(shared_datadir: Path) -> Path:
    return shared_datadir / "example.in"


class TestSuiteGlide:
    @pytest.mark.needs_node("glide")
    def test_Glide(
        self, temp_working_dir: Path, test_config: Config, example_smiles: list[str], grid: Path
    ) -> None:
        rig = TestRig(Glide, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in example_smiles]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(inputs={"inp": [inputs], "inp_grid": [grid]})
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == len(example_smiles)
        confs = mols[0].molecules[0].conformers
        assert confs[0].primary_score < confs[1].primary_score < confs[2].primary_score
        assert mols[0].molecules[0].n_conformers == 4
        assert -8 < mols[0].molecules[0].primary_score < -5
        assert mols[0].n_isomers == 1

    @pytest.mark.needs_node("glide")
    def test_Glide_constraints(
        self,
        temp_working_dir: Path,
        test_config: Config,
        example_smiles: list[str],
        grid_constraints: Path,
        constraint_input: Path,
    ) -> None:
        rig = TestRig(Glide, config=test_config)
        inputs = [IsomerCollection.from_smiles(smi) for smi in example_smiles]
        for inp in inputs:
            inp.embed()

        res = rig.setup_run(
            inputs={"inp": [inputs], "inp_grid": [grid_constraints]},
            parameters={"constraints": constraint_input, "n_jobs": 4},
        )
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == len(example_smiles)
        assert mols[0].molecules[0].n_conformers == 4
        assert -8 < mols[0].molecules[0].primary_score < -5
        assert mols[0].n_isomers == 1
