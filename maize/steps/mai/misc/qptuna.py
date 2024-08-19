"""Qptuna support"""

# pylint: disable=import-outside-toplevel, import-error
import functools
import json
import logging
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, cast

import numpy as np
import pandas as pd
import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output, FileParameter, Suffix, Parameter, Flag
from maize.utilities.testing import TestRig
from maize.utilities.chem import IsomerCollection, Isomer
from maize.utilities.io import Config


log = logging.getLogger("run")


SMILES_COLUMN = "smiles"
TARGET_COLUMN = "score"
PREDICTED_COLUMN = "Prediction"
DEFAULT_DATAPATH = Path("data.csv")
DEFAULT_PATCHED_CONFIG = Path("config-patched.json")


def _mols_to_csv(path: Path, mols: list[IsomerCollection], max_size: int = 1000) -> None:
    """Save SMILES - score information to a CSV file, or update one if it exists"""
    new = pd.DataFrame(
        {
            SMILES_COLUMN: [mol.smiles for mol in mols],
            TARGET_COLUMN: [mol.primary_score for mol in mols],
        }
    )
    if path.exists():
        old = pd.read_csv(path)
        data = pd.concat([old, new]).drop_duplicates(subset=SMILES_COLUMN)
        log.info("Updating pool with %s molecules, %s total", len(new), len(data))
    else:
        data = new
    log.info("Sending %s molecules to Qptuna", len(data[-max_size:]))
    data[-max_size:].to_csv(path)


def _parse_score(isomer: Isomer, value: Any, agg: Literal["min", "max"] = "min") -> Isomer:
    isomer.add_score("qptuna", value, agg=agg)
    isomer.set_tag("score_type", "surrogate")
    isomer.primary_score_tag = "qptuna"
    return isomer


def _parse_unc(isomer: Isomer, value: Any, agg: Literal["min", "max"] = "min") -> Isomer:
    isomer.set_tag("uncertainty", value)
    return isomer


def _csv_to_mols(
    path: Path, mols: list[IsomerCollection], parser: Callable[[Isomer, Any], Isomer]
) -> list[IsomerCollection]:
    """Read qptuna CSV output into existing molecules"""
    data = pd.read_csv(path)
    if any(col not in data.columns for col in (SMILES_COLUMN, TARGET_COLUMN)):
        raise KeyError(
            (f"Could not find columns '{SMILES_COLUMN}' or " f"'{TARGET_COLUMN}' in qptuna output")
        )

    for mol, score in zip(mols, data[PREDICTED_COLUMN]):
        for isomer in mol.molecules:
            parser(isomer, score)
    return mols


def _patch_config(path: Path, training_data: Path) -> Path:
    """Patch the qptuna config 'data' section"""
    with path.open() as file:
        config = json.load(file)
    config["data"] = {
        "training_dataset_file": training_data.as_posix(),
        "input_column": SMILES_COLUMN,
        "response_column": TARGET_COLUMN,
    }
    with DEFAULT_PATCHED_CONFIG.open("w") as file:
        json.dump(config, file)
    return DEFAULT_PATCHED_CONFIG


def _log_results(path: Path, logger: logging.Logger) -> None:
    """Formats the results of hyperparameter optimization"""
    with path.open() as file:
        config = json.load(file)
    logger.info("Best value: %s", config["metadata"]["best_value"])
    logger.info("Best model: %s", config["algorithm"]["name"])
    for key, val in config["algorithm"]["parameters"].items():
        logger.info("  %s = %s", key, val)


class QptunaTrain(Node):
    """
    Interface to Qptuna training.

    Notes
    -----
    See the `Qptuna repo <https://github.com/MolecularAI/Qptuna>`_
    for installation instructions. For maize to access it, specify the python interpreter
    and script location (most likely your python environment ``bin`` folder).

    """

    required_callables = ["qptuna-build"]
    """Requires the 'qptuna-build' callable"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to train the model with"""

    inp_config: Input[Annotated[Path, Suffix("json")]] = Input(cached=True)
    """Qptuna model configuration (``buildconfig``)"""

    out: Output[bool] = Output()
    """Signal emitted for completed training"""

    model: FileParameter[Annotated[Path, Suffix(".pkl")]] = FileParameter(exist_required=False)
    """Merged model output"""

    pool: FileParameter[Annotated[Path, Suffix(".csv")]] = FileParameter(exist_required=False)
    """File to pool training molecules in"""

    n_train: Parameter[int] = Parameter(default=1000)
    """Number of molecules to train with"""

    def run(self) -> None:
        # Make sure we have a consistent 'data' section
        build_config = self.inp_config.receive()
        self.logger.debug("Received config")
        config = _patch_config(build_config, self.pool.filepath)

        # Outputs
        best_model = Path("best.pkl").absolute()

        mols = self.inp.receive()
        self.logger.info("Updating pool with %s molecules", len(mols))
        _mols_to_csv(self.pool.filepath, mols, max_size=self.n_train.value)
        command = (
            f"{self.runnable['qptuna-build']} "
            f"--config {config.as_posix()} "
            f"--best-model-outpath {best_model.as_posix()} "
            f"--merged-model-outpath {self.model.filepath.absolute().as_posix()}"
        )
        self.run_command(command, verbose=True, pre_execution="unset RDBASE")
        self.logger.info("Finished training")
        self.out.send(True)


class QptunaHyper(Node):
    """
    Interface to Qptuna hyperparameter optimisation.

    Notes
    -----
    See the `Qptuna repo <https://github.com/MolecularAI/Qptuna>`_
    for installation instructions. For maize to access it, specify the python interpreter
    and script location (most likely your python environment ``bin`` folder).

    """

    required_callables = ["qptuna-optimize"]
    """Requires the 'qptuna-optimize' callable"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to train the model with"""

    out: Output[Annotated[Path, Suffix("json")]] = Output()
    """Optimised hyperparameter config"""

    configuration: FileParameter[Annotated[Path, Suffix(".json")]] = FileParameter()
    """Qptuna configuration template"""

    model: FileParameter[Annotated[Path, Suffix(".pkl")]] = FileParameter(exist_required=False)
    """Merged model output"""

    pool: FileParameter[Annotated[Path, Suffix(".csv")]] = FileParameter(exist_required=False)
    """File to pool training molecules in"""

    n_train: Parameter[int] = Parameter(default=1000)
    """Number of molecules to optimize with"""

    def run(self) -> None:
        # Make sure we have a consistent 'data' section
        config = _patch_config(self.configuration.filepath, self.pool.filepath)

        # Outputs
        build_out = Path("build.json").absolute()
        best_model = Path("best.pkl").absolute()

        mols = self.inp.receive()
        self.logger.info("Updating pool with %s molecules", len(mols))
        _mols_to_csv(self.pool.filepath, mols, max_size=self.n_train.value)
        command = (
            f"{self.runnable['qptuna-optimize']} "
            f"--config {config.as_posix()} "
            f"--best-buildconfig-outpath {build_out.as_posix()} "
            f"--best-model-outpath {best_model.as_posix()} "
            f"--merged-model-outpath {self.model.filepath.absolute().as_posix()}"
        )
        self.run_command(command, verbose=True, pre_execution="unset RDBASE")
        self.logger.info("Finished hyperparameter optimization")
        _log_results(build_out, self.logger)
        self.out.send(build_out)


# TODO Add `SmilesAndSideInfoFromFile` and / or `PrecomputedDescriptorFromFile`
class QptunaPredict(Node):
    """
    Interface to Qptuna prediction.

    Notes
    -----
    See the `Qptuna repo <https://github.com/MolecularAI/Qptuna>`_
    for installation instructions. For maize to access it, specify the python interpreter
    and script location (most likely your python environment ``bin`` folder).

    """

    required_callables = ["qptuna-predict"]
    """Requires the 'qptuna-predict' callable"""

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to predict using the built model"""

    out: Output[list[IsomerCollection]] = Output()
    """List of tagged molecules with predicted values"""

    model: FileParameter[Annotated[Path, Suffix(".pkl")]] = FileParameter(exist_required=False)
    """Merged model output / reading location"""

    uncertainty: Flag = Flag(default=False)
    """Whether to additionally predict uncertainties (not available for all models)"""

    agg: Parameter[Literal["min", "max"]] = Parameter(default="min")
    """
    What kind of aggregation to use for the score, i.e. ``min``
    if lower scores are better, ``max`` if higher scores are better

    """

    def run(self) -> None:
        data = DEFAULT_DATAPATH.absolute()

        # Outputs
        pred = Path("pred.csv").absolute()

        if self.inp.ready() and not self.model.filepath.exists():
            self.logger.warning("Received data for prediction, but model is not trained yet")
            return
        mols = self.inp.receive()
        _mols_to_csv(data, mols, max_size=len(mols))
        self.logger.info("Predicting %s molecules", len(mols))
        command = (
            f"{self.runnable['qptuna-predict']} "
            f"--model-file {self.model.filepath.absolute().as_posix()} "
            f"--input-smiles-csv-file {data.as_posix()} "
            f"--input-smiles-csv-column {SMILES_COLUMN} "
            f"--output-prediction-csv-file {pred.as_posix()} "
        )
        self.run_command(command, verbose=True, pre_execution="unset RDBASE")
        parser = functools.partial(_parse_score, agg=self.agg.value)
        mols = _csv_to_mols(pred, mols, parser=parser)

        if self.uncertainty.value:
            self.logger.info("Predicting uncertainty for %s molecules", len(mols))
            command += "--predict-uncertainty"
            self.run_command(command, verbose=True, pre_execution="unset RDBASE")
            mols = _csv_to_mols(pred, mols, parser=_parse_unc)

        self.logger.info("Sending predictions for %s molecules", len(mols))
        self.out.send(mols)


@pytest.fixture
def qptuna_example_config(shared_datadir: Path) -> Path:
    return shared_datadir / "qptuna.json"


@pytest.fixture
def qptuna_example_config_mapie(shared_datadir: Path) -> Path:
    return shared_datadir / "build-mapie-best.json"


@pytest.fixture
def qptuna_pool(tmp_path: Path) -> Path:
    return tmp_path / "pool.csv"


# 1UYD ligands (IcolosData)
@pytest.fixture
def train_smiles() -> list[str]:
    return [
        "Nc1ncnc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cc(OC)c(OC)c(c3)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC",
        "Nc1nc(F)nc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc(cc3)cc(c34)OCO4",
        "Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC",
        "Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc(cc3)cc(c34)OCO4",
        "Nc1nc(F)nc(c12)n(CCCC#C)c(n2)Cc3cc(OC)ccc3OC",
    ]


@pytest.fixture
def test_smiles() -> list[str]:
    return [
        "CC(C)NCCCn(c(c12)nc(F)nc2N)c(n1)Cc(c3)c(I)cc(c34)OCO4",
        "CC(C)NCCCn(c(c12)ncnc2N)c(n1)Sc(c3)c(Br)cc(c34)OCO4",
        "CC(C)NCCCn(c(c12)ncnc2N)c(n1)Sc(c3)c(I)cc(c34)OCO4",
        "COc1ccc(OC)c(c1)Cc(n2)[nH]c(c23)c(N)nc(n3)F",
        "O1COc(c12)cc(Br)c(c2)Cc(nc(n34)c(N)ncc3)c4NCc5ccccc5",
    ]


class TestSuiteQptuna:
    @pytest.mark.needs_node("qptunatrain")
    def test_qptuna(
        self,
        qptuna_example_config: Path,
        train_smiles: list[str],
        test_smiles: list[str],
        test_config: Config,
        temp_working_dir: Path,
        qptuna_pool: Path,
    ) -> None:
        """Test our step in isolation"""
        model = Path("model.pkl").absolute()
        mols_train = [IsomerCollection.from_smiles(smi) for smi in train_smiles]
        for mol in mols_train:
            for isomer in mol.molecules:
                isomer.add_score("train", -10 * np.random.random(10))
        mols_test = [IsomerCollection.from_smiles(smi) for smi in test_smiles]

        # Hyperparam
        rig = TestRig(QptunaHyper, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [mols_train]},
            parameters={
                "configuration": qptuna_example_config,
                "model": model,
                "pool": qptuna_pool,
            },
        )
        build_conf = res["out"].get()
        assert build_conf is not None and build_conf.exists()

        # Training
        rig = TestRig(QptunaTrain, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [mols_train], "inp_config": [build_conf]},
            parameters={"model": model, "pool": qptuna_pool},
        )
        assert res["out"].get()

        rig = TestRig(QptunaPredict, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [mols_test]},
            parameters={"model": model, "uncertainty": False},
        )
        mols = res["out"].get()
        assert mols is not None
        assert hasattr(mols, "__len__")
        assert len(mols) == 5
        for mol in mols:
            assert mol.scored
            for iso in mol.molecules:
                assert iso.scores is not None
                assert -12 < iso.scores["qptuna"] < 2

    @pytest.mark.xfail(reason="Recent change in Qptuna schema for uncertainty prediction")
    @pytest.mark.needs_node("qptunatrain")
    def test_qptuna_unc(
        self,
        qptuna_example_config_mapie: Path,
        train_smiles: list[str],
        test_smiles: list[str],
        test_config: Config,
        temp_working_dir: Path,
        qptuna_pool: Path,
    ) -> None:
        """Test our step in isolation"""
        model = Path("model.pkl").absolute()
        mols_train = [IsomerCollection.from_smiles(smi) for smi in train_smiles]
        for mol in mols_train:
            for isomer in mol.molecules:
                isomer.add_score("train", -10 * np.random.random(10))
        mols_test = [IsomerCollection.from_smiles(smi) for smi in test_smiles]

        # Training
        rig = TestRig(QptunaTrain, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [mols_train], "inp_config": [qptuna_example_config_mapie]},
            parameters={"model": model, "pool": qptuna_pool},
        )
        assert res["out"].get()

        rig = TestRig(QptunaPredict, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [mols_test]},
            parameters={"model": model, "uncertainty": True},
        )
        mols = res["out"].get()
        assert mols is not None
        assert hasattr(mols, "__len__")
        assert len(mols) == 5
        for mol in mols:
            assert mol.scored
            for iso in mol.molecules:
                assert iso.scores is not None
                assert -12 < iso.scores["qptuna"] < 2
                assert iso.has_tag("uncertainty")
                assert 0 < float(cast(float, iso.get_tag("uncertainty"))) < 10
