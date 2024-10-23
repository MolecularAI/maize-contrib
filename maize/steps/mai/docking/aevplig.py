"""Runs the AEV-PLIG protein-ligand pose scorer"""

import csv
from pathlib import Path
from shutil import copytree
from typing import Annotated

import numpy as np
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, FileParameter, Suffix, Input, Output
from maize.utilities.chem import Isomer, IsomerCollection
from maize.utilities.testing import TestRig
from maize.utilities.io import Config
from maize.utilities.utilities import unique_id
from maize.utilities.validation import FileValidator


class AEVPLIG(Node):
    """
    Runs AEV-PLIG on a ligand pose. This is a scoring function that takes an
    existing ligand pose (obtained from e.g. docking) with the corresponding
    receptor conformation and outputs a predicted binding affinity. See the
    [full code](https://github.com/isakvals/AEV-PLIG) for details.

    References
    ----------
    .. [#warren2024] Warren, M. T., Valsson, I., Deane, C. M., Magarkar, A.,
       Morris, G. M. & Biggin, P. C. How to make machine learning scoring
       functions competitive with FEP. ChemRxiv (2024)

    """
    tags = {"chemistry", "scorer", "tagger"}

    required_callables = ["aevplig"]
    """
    aevplig
        The AEV-PLIG `process_and_predict.py` script

    """

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    receptor: FileParameter[Annotated[Path, Suffix("pdb")]] = FileParameter()
    """Protein structure"""

    model: Parameter[str] = Parameter()
    """Trained model to use for scoring"""

    aev_source: FileParameter[Path] = FileParameter()
    """Source folder location for AEV-PLIG"""

    def run(self) -> None:
        mols = self.inp.receive()
        protein = self.receptor.filepath

        # A quirk in AEV-PLIG doesn't allow a single molecule to be parsed,
        # so we do the hacky thing and duplicate it (and remove it later)
        single_mol = len([iso for mol in mols for iso in mol.molecules]) == 1
        if single_mol:
            mols = [mols[0], mols[0]]

        run_dir = Path("run")
        run_dir.mkdir()
        copytree(self.aev_source.value / "data", run_dir / "data", ignore_dangling_symlinks=True)
        copytree(
            self.aev_source.value / "output", run_dir / "output", ignore_dangling_symlinks=True
        )

        table = run_dir / "input.csv"
        isos: dict[str, Isomer] = {}
        with table.open("w") as out:
            writer = csv.writer(out, delimiter=",")
            writer.writerow(["unique_id", "pK", "sdf_file", "pdb_file"])
            for mol in mols:
                for iso in mol.molecules:
                    name = unique_id(6)
                    input_file = run_dir / f"inp-{iso.inchi}.sdf"
                    iso.to_sdf(input_file)
                    writer.writerow(
                        [name, 0.0, input_file.absolute().as_posix(), protein.absolute().as_posix()]
                    )
                    isos[name] = iso

        command = (
            f"{self.runnable['aevplig']} --dataset_csv={table.absolute().as_posix()} "
            f"--data_name=maize --trained_model_name={self.model.value}"
        )
        output_dir = run_dir / "output" / "predictions"
        output_file = output_dir / "maize_predictions.csv"
        self.run_command(
            command,
            working_dir=run_dir,
            validators=[FileValidator(output_file)],
            raise_on_failure=False,
        )

        parsed = set()
        with output_file.open("r") as inp:
            reader = csv.reader(inp)
            _ = next(reader)
            for name, _, _, _, _, *predictions, average in reader:
                iso = isos[name]
                std = np.array(predictions, dtype=float).std()
                iso.add_score("aev-plig", float(average), agg="max")
                iso.set_tag("aev-plig", float(average))
                iso.set_tag("aev-plig-std", std)
                self.logger.info(
                    "Predicted affinity (pK) for %s: %s (Std: %s)",
                    iso.name or iso.inchi,
                    float(average),
                    std,
                )
                parsed.add(name)

        for name, iso in isos.items():
            if name not in parsed:
                self.logger.warning("Prediction failed for %s", iso.name or iso.inchi)
                iso.add_score("aev-plig", np.nan, agg="max")
                iso.set_tag("aev-plig-std", 0.0)

        if single_mol:
            mols = [mols[0]]
        self.out.send(mols)


@pytest.fixture
def protein(shared_datadir: Path) -> Annotated[Path, Suffix("pdb")]:
    return shared_datadir / "1UYD_apo.pdb"


@pytest.fixture
def mol(shared_datadir: Path) -> Isomer:
    return Isomer.from_sdf(shared_datadir / "1UYD_ligand.sdf")


class TestSuite_AEVPLIG:
    @pytest.mark.needs_node("aevplig")
    def test_AEVPLIG(
        self, temp_working_dir: Path, test_config: Config, protein: Path, mol: Isomer
    ) -> None:
        rig = TestRig(AEVPLIG, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[IsomerCollection([mol]), IsomerCollection([mol])]]},
            parameters={"receptor": protein},
        )
        out = res["out"].get()
        assert out is not None
        assert len(out) == 2
        assert "aev-plig" in out[0].molecules[0].scores
        assert out[0].molecules[0].scores["aev-plig"] > 2.0

    @pytest.mark.needs_node("aevplig")
    def test_AEVPLIG_single(
        self, temp_working_dir: Path, test_config: Config, protein: Path, mol: Isomer
    ) -> None:
        rig = TestRig(AEVPLIG, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[IsomerCollection([mol])]]}, parameters={"receptor": protein}
        )
        out = res["out"].get()
        assert out is not None
        assert len(out) == 1
        assert "aev-plig" in out[0].molecules[0].scores
        assert out[0].molecules[0].scores["aev-plig"] > 2.0
