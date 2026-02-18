"""Schrodinger Ligprep prepares 3D small molecule conformers and isomers"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Literal, cast

import pytest

from maize.core.interface import Input, Output, Parameter, Flag, FileParameter
from maize.utilities.testing import TestRig
from maize.utilities.utilities import unique_id

from maize.steps.mai.common.schrodinger import Schrodinger
from maize.utilities.chem import IsomerCollection, load_sdf_library, save_sdf_library
from maize.utilities.io import Config, wait_for_file


LigPrepKeywordsType = dict[str, str | int | float]


class Ligprep(Schrodinger):
    """
    Calls Schrodinger's Ligprep tool to embed small molecules and create isomers.

    Notes
    -----
    Due to Schrodinger's licensing system, each call to a tool requires
    going through Schrodinger's job server. This is run separately for
    each job to avoid conflicts with a potentially running main server.

    See Also
    --------
    :class:`~maize.steps.mai.molecule.Smiles2Molecules` :
        A simple, fast, and less accurate alternative to
        Gypsum and Ligprep, using RDKit embedding functionality.
    :class:`~maize.steps.mai.molecule.Gypsum` :
        A more advanced procedure for producing different isomers and
        high-energy conformers, and an open-source alternative to ligprep.

    """

    tags = {"chemistry", "embedding"}

    N_LICENSES = 1

    required_callables = Schrodinger.required_callables + ["ligprep"]

    inp: Input[list[str] | list[IsomerCollection]] = Input()
    """SMILES or Maize IsomerCollection objects as input"""

    out: Output[list[IsomerCollection]] = Output()
    """Embedded isomer collection output"""

    epik: Flag = Flag(default=True)
    """Whether to use Epik Classic for ionization and tautomerization"""

    epikx: Flag = Flag(default=False)
    """Whether to use EpikX for ionization and tautomerization. Takes precedence over epik if both are set."""

    ionization: Parameter[Literal[0, 1, 2]] = Parameter(optional=True)
    """Ionization treatment; 0 - do not ionize / neutralize, 1 - only neutralize, 2 - both (redundant if epik is enabled)"""

    skip_tautomerization: Flag = Flag(default=False)
    """Skip tautomerization (adds -nt flag). Overridden by epik or ionization if those are set."""

    ph: Parameter[float] = Parameter(optional=True)
    """Target pH"""

    ph_tolerance: Parameter[float] = Parameter(optional=True)
    """pH tolerance"""

    max_stereo: Parameter[int] = Parameter(default=32)
    """Maximum number of stereoisomers to generate"""

    ignore_chiralities: Flag = Flag(default=False)
    """Respect chiralities from input geometry when generating stereoisomers"""

    lig_filter_file: FileParameter[Path] = FileParameter(optional=True)
    """An optional LigFilter file to filter certain structures"""

    keywords: Parameter[LigPrepKeywordsType] = Parameter(default_factory=dict)
    """
    Additional LigPrep keywords to use, see the LigPrep command help (there is no online documentation, good luck).

    """

    def run(self) -> None:
        input_list = self.inp.receive()
        mols = []

        if len(input_list) == 0:
            self.out.send(mols)
            return

        elif isinstance(input_list[0], IsomerCollection):
            # input is Isomercollection
            # This is needed to shutup mypy
            input_list = cast(list[IsomerCollection], input_list)
            for i, mol in enumerate(input_list):
                for iso in mol.molecules:
                    iso.set_tag("i_sm_idx", i)
                mols.append(mol)

            input_file_path = Path("input.sdf")
            save_sdf_library(input_file_path, input_list, split_strategy="schrodinger")
            in_format_flag = "isd"

        else:
            # SMILES input, we save as CSV to attach the original index and retrieve it later
            input_list = cast(list[str], input_list)
            input_file_path = Path(
                "input.csv"
            )  # MUST have CSV extension, otherwise Ligprep will crash
            with input_file_path.open("w") as out:
                out.write("smiles,idx\n")
                for i, smi in enumerate(input_list):
                    out.write(f"{smi.strip()},{i}\n")
            in_format_flag = "icsv"
            # we need to make an iso collection list anyway so that the output list merges successes and failures consistently
            for i, inp in enumerate(input_list):
                mol = IsomerCollection.from_smiles(inp)
                mol.set_tag("original_smiles", inp)
                for iso in mol.molecules:
                    iso.set_tag("i_sm_idx", i)
                mols.append(mol)

        self.logger.debug(f"entering Ligprep, input list is {len(input_list)} elements long")

        name = f"ligprep-{unique_id(12)}"
        output_sdf = Path(f"{name}.sdf")
        # While it would be enticing to add '-LOCAL' here, this will
        # cause a DeprecationWarning that actually crashes the program :(
        command = (
            f"{self.runnable['ligprep']} -{in_format_flag} {input_file_path.as_posix()} "
            f"-osd {output_sdf.as_posix()} "
            f"-s {self.max_stereo.value} "
        )
        # Check for conflicting epik settings
        if self.epik.value and self.epikx.value:
            self.logger.warning(
                "Both 'epik' and 'epikx' are set. Using 'epikx' only as they cannot be used together."
            )

        if self.ionization.is_set and (self.epik.value or self.epikx.value):
            self.logger.warning(
                "Both 'ionization' and 'epik'/'epikx' are set. The 'ionization' parameter will be ignored "
                "as Epik handles ionization treatment."
            )
        if self.skip_tautomerization.value:
            self.logger.info(
                "'skip_tautomerization' is set."
            )
        if self.ionization.is_set:
            command += f"-i {self.ionization.value} "
        # Use epikx if set, otherwise use epik (but not both), else ligprep fails with error
        if self.epikx.value:
            command += "-epikx "
        elif self.epik.value:
            command += "-epik "
        if self.skip_tautomerization.value:
            command += "-nt "
        if self.ph.is_set:
            command += f"-ph {self.ph.value} "
        if self.ph_tolerance.is_set:
            command += f"-pht {self.ph_tolerance.value} "
        if not self.ignore_chiralities.value:
            command += "-g "
        if self.lig_filter_file.is_set:
            command += f"-f {self.lig_filter_file.filepath.as_posix()} "

        if self.keywords.is_set:
            arg_value_diction = self.keywords.value
            for key in arg_value_diction:
                command += f"-{key} {arg_value_diction[key]} "

        self._run_schrodinger_job(command, name=name, verbose=True)
        wait_for_file(output_sdf, timeout=5)
        parsed_mols = load_sdf_library(output_sdf, split_strategy="schrodinger-tag")

        # Overwrite input list with successfully generated mols
        for mol in parsed_mols:
            original_index = int(cast(int, mol.molecules[0].get_tag("i_sm_idx")))
            mols[original_index] = mol

        self.logger.debug(f"exiting Ligprep, output list is {len(mols)} elements long")

        self.out.send(mols)


@pytest.fixture
def sdf_input(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_ligands.sdf"


@pytest.fixture
def protonation_test_input(shared_datadir: Path) -> Path:
    return shared_datadir / "fluorophenol.sdf"

@pytest.fixture
def tautomer_input() -> list[str]:
    # pyridone/hydroxypyridine tautomer pair
    return ["Oc1ccccn1"]

class TestSuiteLigprep:
    @pytest.mark.needs_node("ligprep")
    def test_Ligprep(
        self,
        temp_working_dir: Path,
        test_config: Config,
        example_smiles: list[str],
        sdf_input: Path,
        protonation_test_input: Path,
    ) -> None:
        rig = TestRig(Ligprep, config=test_config)
        res = rig.setup_run(inputs={"inp": [example_smiles]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == len(example_smiles)
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].n_isomers == 1

        isoc = load_sdf_library(sdf_input, split_strategy="none")
        res = rig.setup_run(inputs={"inp": [isoc]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].n_isomers == 1
        assert mols[1].n_isomers == 1
        assert mols[2].n_isomers == 1

        isoc = load_sdf_library(protonation_test_input, split_strategy="none")
        res = rig.setup_run(inputs={"inp": [isoc]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 1
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].n_isomers == 2

    @pytest.mark.needs_node("ligprep")
    def test_Ligprep_bad(
        self,
        temp_working_dir: Path,
        test_config: Config,
        evil_example_smiles: list[str],
    ) -> None:
        rig = TestRig(Ligprep, config=test_config)
        res = rig.setup_run(inputs={"inp": [evil_example_smiles]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == len(evil_example_smiles)
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].n_isomers == 1
        assert mols[-1].n_isomers == 0

    @pytest.mark.needs_node("ligprep")
    def test_Ligprep_epikx(
        self,
        temp_working_dir: Path,
        test_config: Config,
        tautomer_input: list[str],
    ) -> None:
        from rdkit import Chem

        """Test using EpikX instead of Epik"""
        rig = TestRig(Ligprep, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [tautomer_input]},
            parameters={"epik": False, "epikx": True}
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 1
        assert not mols[0].molecules[0]._molecule.HasSubstructMatch(Chem.MolFromSmarts("[OH]"))


    @pytest.mark.needs_node("ligprep")
    def test_Ligprep_skip_tautomerization(
        self,
        temp_working_dir: Path,
        test_config: Config,
        tautomer_input: list[str],
    ) -> None:
        from rdkit import Chem
        """Test skip_tautomerization flag"""
        rig = TestRig(Ligprep, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [tautomer_input]},
            parameters={"epik": False, "skip_tautomerization": True}
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 1
        assert mols[0].molecules[0]._molecule.HasSubstructMatch(Chem.MolFromSmarts("[OH]"))
