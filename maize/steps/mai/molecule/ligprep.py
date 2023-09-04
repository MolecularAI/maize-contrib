"""Schrodinger Ligprep prepares 3D small molecule conformers and isomers"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Any, Literal

import pytest

from maize.core.interface import Input, Output, Parameter, Flag
from maize.utilities.testing import TestRig
from maize.utilities.validation import SuccessValidator

from maize.steps.mai.common.schrodinger import Schrodinger, has_license
from maize.utilities.chem import IsomerCollection, save_smiles, load_sdf_library


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

    required_callables = ["ligprep"]

    inp: Input[list[str]] = Input()
    """SMILES input"""

    out: Output[list[IsomerCollection]] = Output()
    """Embedded isomer collection output"""

    epik: Flag = Flag(default=True)
    """Whether to use Epik for ionization and tautomerization"""

    ionization: Parameter[Literal[0, 1, 2]] = Parameter(default=1)
    """Ionization treatment: 0 - do not ionize / neutralize, 1 - only neutralize, 2 - both"""

    ph: Parameter[float] = Parameter(optional=True)
    """Target pH"""

    ph_tolerance: Parameter[float] = Parameter(optional=True)
    """pH tolerance"""

    max_stereo: Parameter[int] = Parameter(default=32)
    """Maximum number of stereoisomers to generate"""

    def run(self) -> None:
        smiles = [smi.strip() for smi in self.inp.receive()]
        smiles_path = Path("input.smi")
        output_sdf = Path("output.sdf")
        save_smiles(smiles_path, smiles)

        # While it would be enticing to add '-LOCAL' here, this will
        # cause a DeprecationWarning that actually crashes the program :(
        command = (
            f"{self.runnable['ligprep']} -ismi {smiles_path.as_posix()} "
            f"-osd {output_sdf.as_posix()} -i {self.ionization.value} "
            f"-s {self.max_stereo.value} -NJOBS {self.n_jobs.value} "
            f"-WAIT -HOST {self.host.value} "
        )
        if self.epik.value:
            command += "-epik "
        if self.ph.is_set:
            command += f"-ph {self.ph.value} "
        if self.ph_tolerance.is_set:
            command += f"-pht {self.ph_tolerance.value} "

        self.run_command(command, validators=[SuccessValidator("JobId:")])
        mols = load_sdf_library(output_sdf, split_strategy="schrodinger-tag")
        self.out.send(mols)


@pytest.mark.skipif(not has_license(), reason="No Schrodinger license found")
class TestSuiteLigprep:
    def test_Ligprep(self, temp_working_dir: Any, test_config: Any, example_smiles: Any) -> None:
        rig = TestRig(Ligprep, config=test_config)
        res = rig.setup_run(inputs={"inp": [example_smiles]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 5
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].n_isomers == 1
