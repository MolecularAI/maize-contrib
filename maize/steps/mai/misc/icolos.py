"""Legacy icolos support"""

# pylint: disable=import-outside-toplevel, import-error
import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output, FileParameter, Suffix
from maize.utilities.testing import TestRig
from maize.utilities.chem import IsomerCollection, save_sdf_library
from maize.utilities.io import Config


NAME_COLUMN = "_Name"
TARGET_COLUMN = "ddG"


def _csv_to_mols(path: Path, mols: list[IsomerCollection]) -> list[IsomerCollection]:
    """Read icolos CSV output into existing molecules"""
    data = pd.read_csv(path)
    if any(col not in data.columns for col in (NAME_COLUMN, TARGET_COLUMN)):
        raise KeyError(
            (f"Could not find columns '{NAME_COLUMN}' or " f"'{TARGET_COLUMN}' in Icolos output")
        )

    mols_dict = {mol.molecules[0].name: mol for mol in mols}
    for name, score in zip(data[NAME_COLUMN], data[TARGET_COLUMN]):
        isomer = mols_dict[name].molecules[0]
        isomer.set_tag("ddg", score)
        isomer.set_tag("score_type", "surrogate")
        isomer.add_score_tag("ddg")
    return mols


def _patch_icolos_conf(inp: Path, out: Path, global_variables: dict[str, str]) -> None:
    """Adds global variables to an Icolos config"""
    with inp.open() as conf:
        data = json.load(conf)
    data["workflow"]["header"]["global_variables"] = global_variables
    with out.open("w") as output:
        json.dump(data, output, indent=4)


class IcolosFEP(Node):
    """
    Interface to Icolos scoring functions for molecules.

    Notes
    -----
    See the `Icolos repo <https://github.com/MolecularAI/Icolos>`_
    for installation instructions. For maize to access it, specify the python interpreter
    and icolos executable location (most likely your python environment ``bin`` folder).

    """

    required_callables = ["icolos"]
    """
    scripts
        Requires the ``icolos`` executable and a suitable python interpreter
    environment
        Requires the following environment variables to be set:

        * ``PMX_PYTHON``
            
          Path to the python executable for PMX

        * ``PMX``
        
          Path to the PMX executable

        * ``GMXLIB``

          Path to the PMX mutff force field
        
        * ``ICOLOS_ENTRY``

          Path to the ``icolos_entrypoints`` PMX script folder

    """

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to send to Icolos"""

    inp_reference: Input[IsomerCollection] = Input()
    """Reference molecule for the star-map"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with attached scores"""

    target: FileParameter[Annotated[Path, Suffix(".pdb")]] = FileParameter()
    """Target protein structure"""

    configuration: FileParameter[Annotated[Path, Suffix(".json")]] = FileParameter()
    """Icolos configuration"""

    mdps: FileParameter[Path] = FileParameter()
    """Folder containing mdp files for Gromacs"""

    def run(self) -> None:
        mols = self.inp.receive()
        ref = self.inp_reference.receive()
        mols.append(ref)

        sdf_in = Path("input.sdf")
        result_file = Path("output.csv")
        conf = Path("icolos.json")
        for mol in mols:
            self.logger.debug("Saving mol '%s' to SDF", mol.molecules[0].name)

        save_sdf_library(sdf_in, mols, split_strategy="none")
        global_variables = {
            "compounds": sdf_in.absolute().as_posix(),
            "target": self.target.filepath.absolute().as_posix(),
            "output": self.work_dir.absolute().as_posix(),
            "results": result_file.absolute().as_posix(),
            "mdps": self.mdps.filepath.absolute().as_posix(),
            "reference_name": ref.molecules[0].name or "ref",
        }
        _patch_icolos_conf(
            inp=self.configuration.filepath, out=conf, global_variables=global_variables
        )
        command = f"{self.runnable['icolos']} -debug " f"-conf {conf.absolute().as_posix()}"
        self.run_command(command, verbose=True)
        _csv_to_mols(result_file, mols)
        self.out.send(mols)


@pytest.fixture
def icolos_example_config(shared_datadir: Path) -> Path:
    return shared_datadir / "dockpose2rbfe_gpu.json"


@pytest.fixture
def icolos_target(shared_datadir: Path) -> Path:
    return shared_datadir / "1stp_protein.pdb"


@pytest.fixture
def icolos_mdps(shared_datadir: Path) -> Path:
    return shared_datadir / "mdps"


@pytest.mark.skip(reason="Incomplete implementation")
def test_icolos(
    icolos_example_config: Path,
    example_smiles: list[str],
    test_config: Config,
    icolos_mdps: Path,
    icolos_target: Path,
) -> None:
    """Test our step in isolation"""
    inp_mols = [IsomerCollection.from_smiles(smi) for smi in example_smiles]
    rig = TestRig(IcolosFEP, config=test_config)
    res = rig.setup_run(
        inputs={"inp": [inp_mols[1:]], "inp_reference": [inp_mols[0]]},
        parameters={
            "configuration": icolos_example_config,
            "mdps": icolos_mdps,
            "target": icolos_target,
        },
    )
    mols = res["out"].get()
    assert mols is not None
    assert len(mols) == len(example_smiles)
    for mol in mols:
        assert mol.scored
        for iso in mol.molecules:
            assert iso.scores is not None
            assert -12 < iso.scores[0] < 2
