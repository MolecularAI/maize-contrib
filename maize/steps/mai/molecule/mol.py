"""Molecule handling steps"""

# pylint: disable=import-outside-toplevel, import-error

import json
from pathlib import Path
import random
from typing import Annotated, Any, List, Union

import numpy as np
from numpy.typing import NDArray

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, FileParameter, Suffix, Flag
from maize.utilities.chem.chem import ChemistryException, Isomer, load_sdf_library
from maize.utilities.testing import TestRig

from maize.utilities.chem import IsomerCollection


class Smiles2Molecules(Node):
    """
    Converts SMILES codes into a set of molecules with distinct
    isomers and conformers using the RDKit embedding functionality.

    See Also
    --------
    :class:`~maize.steps.mai.molecule.Gypsum` :
        A more advanced procedure for producing different
        isomers and high-energy conformers.

    """

    inp: Input[list[str]] = Input()
    """SMILES input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    n_conformers: Parameter[int] = Parameter(default=1)
    """Number of conformers to generate"""

    n_variants: Parameter[int] = Parameter(default=1)
    """Maximum number of stereoisomers to generate"""

    embed: Flag = Flag(default=True)
    """
    Whether to create embeddings for the molecule. May not be
    required if passing it on to another embedding system.

    """

    def run(self) -> None:
        smiles = self.inp.receive()
        mols: list[IsomerCollection] = []
        n_variants = self.n_variants.value if self.embed.value else 0
        for i, smi in enumerate(smiles):
            self.logger.info("Embedding %s/%s ('%s')", i + 1, len(smiles), smi.strip())
            try:
                mol = IsomerCollection.from_smiles(smi, max_isomers=n_variants)
                if self.embed.value:
                    mol.embed(self.n_conformers.value)
            except ChemistryException as err:
                self.logger.warning("Unable to create '%s' (%s), not sanitizing...", smi, err)
                if "SMILES Parse Error" in err.args[0]:
                    mol = IsomerCollection([])
                    mol.smiles = smi
                else:
                    mol = IsomerCollection.from_smiles(smi, max_isomers=0, sanitize=False)
            mols.append(mol)
        self.out.send(mols)


class SaveMolecule(Node):
    """Save a molecule to an SDF file."""

    inp: Input[IsomerCollection] = Input()
    """Molecule input"""

    path: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(exist_required=False)
    """SDF output destination"""

    def run(self) -> None:
        mol = self.inp.receive()
        self.logger.info("Received '%s'", mol)
        mol.to_sdf(self.path.value)


class LoadMolecule(Node):
    """Load a molecule from an SDF file."""

    out: Output[Isomer] = Output()
    """Isomer output"""

    path: Input[Annotated[Path, Suffix("sdf")]] = Input()
    """Path to the SDF file"""

    def run(self) -> None:
        input_path = self.path.receive()
        mol = IsomerCollection.from_sdf(input_path)
        self.out.send(mol.molecules[0])

class LoadSmiles(Node):
    """Load SMILES codes from a ``.smi`` file."""

    path: FileParameter[Annotated[Path, Suffix("smi")]] = FileParameter()
    """SMILES file input"""

    out: Output[list[str]] = Output()
    """SMILES output"""

    sample: Parameter[int] = Parameter(optional=True)
    """Take a sample of SMILES"""

    def run(self) -> None:
        with self.path.filepath.open() as file:
            smiles = [smi.strip("\n") for smi in file.readlines()]
            if self.sample.is_set:
                smiles = random.choices(smiles, k=self.sample.value)
            self.out.send(smiles)


class ToSmiles(Node):
    """ transform an isomer or IsomerCollection to SMILES """

    inp: Input[Union[Isomer,IsomerCollection]] = Input()
    """SMILES output"""

    out: Output[List[str]] = Output()
    """SMILES output"""

    def run(self) -> None:
        smiles = self.inp.receive().to_smiles()
        if isinstance(smiles,str): # catch the case where used with single isomer
            smiles = [smiles]
        self.out.send(smiles)

class SaveLibrary(Node):
    """Save a list of molecules to multiple SDF files."""

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule library input"""

    base_path: FileParameter[Path] = FileParameter(exist_required=False)
    """Base output file path name without a suffix, i.e. /path/to/output"""

    def run(self) -> None:
        mols = self.inp.receive()
        base = self.base_path.value
        for i, mol in enumerate(mols):
            file = base.with_name(f"{base.name}{i}.sdf")
            mol.to_sdf(file)


class SaveScores(Node):
    """Save VINA Scores to a JSON file."""

    inp: Input[NDArray[np.float32]] = Input()
    """Molecule input"""

    path: FileParameter[Annotated[Path, Suffix("json")]] = FileParameter(exist_required=False)
    """JSON output destination"""

    def run(self) -> None:
        scores = self.inp.receive()
        self.logger.info(f"Received #{len(scores):d} scores")
        with open(self.path.value, "w") as f:
            json.dump(list(scores), f)


class LoadLibrary(Node):
    """Load a small molecule library from an SDF file"""

    path: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter()
    """Input SDF file"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output, each entry in the SDF is parsed as a separate molecule"""

    def run(self) -> None:
        mols = load_sdf_library(self.path.filepath, split_strategy="none")
        self.out.send(mols)


class TestSuiteMol:
    def test_Smiles2Molecules(self, test_config: Any) -> None:
        rig = TestRig(Smiles2Molecules, config=test_config)
        smiles = ["Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"]
        res = rig.setup_run(
            inputs={"inp": [smiles]}, parameters={"n_conformers": 2, "n_isomers": 2}
        )
        raw = res["out"].get()
        assert raw is not None
        mol = raw[0]
        assert mol.n_isomers <= 2
        assert not mol.scored
        assert mol.molecules[0].n_conformers == 2
        assert mol.molecules[0].charge == 0
        assert mol.molecules[0].n_atoms == 44

    def test_SaveMolecule(self, tmp_path: Path, test_config: Any) -> None:
        rig = TestRig(SaveMolecule, config=test_config)
        mol = IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC")
        rig.setup_run(inputs={"inp": mol}, parameters={"path": tmp_path / "file.sdf"})
        assert (tmp_path / "file.sdf").exists()

    def test_LoadSmiles(self, shared_datadir: Path, test_config: Any) -> None:
        rig = TestRig(LoadSmiles, config=test_config)
        res = rig.setup_run(parameters={"path": shared_datadir / "test.smi"})
        mol = res["out"].get()
        assert mol is not None
        assert len(mol) == 1
        assert mol[0] == "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"

    def test_LoadLibrary(self, shared_datadir: Path, test_config: Any) -> None:
        rig = TestRig(LoadLibrary, config=test_config)
        res = rig.setup_run(parameters={"path": shared_datadir / "1UYD_ligands.sdf"})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3

    def test_SaveLibrary(self, tmp_path: Path, test_config: Any) -> None:
        rig = TestRig(SaveLibrary, config=test_config)
        mols = [
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3")
        ]
        base = tmp_path / "mol"
        rig.setup_run(inputs={"inp": [mols]}, parameters={"base_path": base})
        assert base.with_name("mol0.sdf").exists()
        assert base.with_name("mol1.sdf").exists()
