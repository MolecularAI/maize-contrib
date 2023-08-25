"""Interface to REINVENT"""

from collections.abc import Callable
import json
import os
from pathlib import Path
import stat
import sys
import threading
from time import sleep
from typing import TYPE_CHECKING, Annotated, Any, NoReturn, cast
import pytest

import toml
import numpy as np
from numpy.typing import NDArray

from maize.core.node import Node
from maize.core.workflow import Workflow
from maize.core.interface import Input, Output, Parameter, FileParameter, Suffix, Flag
from maize.steps.io import LoadData, Return
from maize.utilities.chem import IsomerCollection
from maize.utilities.execution import CommandRunner
from maize.utilities.testing import TestRig
from maize.utilities.io import setup_workflow


if TYPE_CHECKING:
    from maize.core.graph import Graph


REINVENT_LOGFILE = Path("reinvent.log")
DEFAULT_PATCHED_CONFIG = Path("config.toml")
TEMP_SMILES_FILE = Path("inp.smi")
TEMP_SCORES_FILE = Path("out.json")
INTERCEPTOR_FILE = Path("./intercept.py")
INTERCEPTOR = f"""#!/usr/bin/env python
from pathlib import Path
import shutil
from time import sleep
import sys

with Path("{TEMP_SMILES_FILE.as_posix()}").open("w") as file:
    file.writelines(sys.stdin.readlines())
while not Path("{TEMP_SCORES_FILE.as_posix()}").exists():
    sleep(0.5)
with Path("{TEMP_SCORES_FILE.as_posix()}").open("r") as file:
    print(file.read())
Path("{TEMP_SCORES_FILE.as_posix()}").unlink()
"""


def expose_reinvent(graph_type: type["Graph"]) -> Callable[[], None]:
    """
    Converts a subgraph with smiles input and score output to a REINVENT-compatible workflow.

    The subgraph must have a single input 'inp' of type `Input[list[str]]`
    and a single output 'out' of type `Output[NDArray[np.float32]]`.

    Parameters
    ----------
    graph_type
        The subgraph to convert

    Returns
    -------
    Callable[[], None]
        Runnable workflow

    """

    def wrapped() -> None:
        flow = Workflow()
        smi = flow.add(LoadData[list[str]], name="smiles")
        core = flow.add(graph_type, name="core")
        sco = flow.add(Return[NDArray[np.float32]])

        assert hasattr(core, "inp")
        assert hasattr(core, "out")

        # Small hack to allow us to use help, but at the
        # same time allow reading in all input from stdin
        if all(flag not in sys.argv for flag in ("-h", "--help")):
            smiles = sys.stdin.readlines()
            flow.nodes
            smi.data.set(smiles)

        flow.connect_all((smi.out, core.inp), (core.out, sco.inp))
        flow.map(*core.all_parameters.values())

        setup_workflow(flow)

        # 1 is stdout, 2 is stderr
        if (scores := sco.get()) is not None:
            print(json.dumps(list(scores)))

    return wrapped


def _exception_handler(args: Any, /) -> NoReturn:
    raise args.exc_type(args.exc_value).with_traceback(args.exc_traceback)


def _write_interceptor(file: Path, contents: str) -> None:
    """Creates a SMILES interceptor script"""
    with file.open("w") as script:
        script.write(contents)
    os.chmod(file, mode=stat.S_IXUSR | stat.S_IRUSR | stat.S_IWUSR)


def _patch_config(
    path: Path,
    weight: float = 1.0,
    low: float = 0.0,
    high: float = 1.0,
    k: float = 0.5,
    reverse: bool = False,
    min_epochs: int = 5,
    max_epochs: int = 10,
    batch_size: int = 128,
    maize_backend: bool = False,
) -> tuple[Path, int]:
    """Patch the REINVENT config to allow interception of SMILES."""
    score_conf = {
        "component_type": "maize" if maize_backend else "external",
        "name": "maize",
        "weight": weight,
        "specific_parameters": {
            "executable": "./intercept.py",
            "transformation": {
                "low": low,
                "high": high,
                "k": k,
                "transformation_type": "reverse_sigmoid" if reverse else "sigmoid",
            },
        },
    }
    with path.open() as file:
        if path.suffix == ".json":
            conf = json.load(file)
            version = conf.get("version", 3)
        elif path.suffix == ".toml":
            conf = toml.load(file)
            version = conf.get("version", 4)
        else:
            raise IOError(f"Unable to read REINVENT config '{path.as_posix()}'")

    if version >= 4:
        conf["stage"][0]["scoring_function"]["component"].append(score_conf)
        conf["stage"][0]["min_steps"] = min_epochs
        conf["stage"][0]["max_steps"] = max_epochs
        conf["parameters"]["batch_size"] = batch_size
    elif version < 4:
        # TODO: min_steps?
        conf["parameters"]["scoring_function"]["parameters"].append(score_conf)
        conf["parameters"]["reinforcement_learning"]["n_steps"] = max_epochs
        conf["parameters"]["reinforcement_learning"]["batch_size"] = batch_size

    patched_file = DEFAULT_PATCHED_CONFIG.with_suffix(".json" if version < 4 else ".toml")
    with patched_file.open("w") as out:
        if version >= 4:
            toml.dump(conf, out)
        else:
            json.dump(conf, out)
    return patched_file, version


class read_log:
    def __init__(self, logfile: Path) -> None:
        self.logfile = logfile
        self._new_lines = 0

    def __call__(self) -> str:
        """Read the ReInvent logfile and format"""
        with self.logfile.open() as log:
            lines = log.readlines()
            all_lines = len(lines)
            msg = "\n"
            msg += "---------------- STDOUT ----------------\n"
            msg += "".join(lines[-(all_lines - self._new_lines) :]) + "\n"
            msg += "---------------- STDOUT ----------------\n"
            self._new_lines = all_lines
        return msg


class ReInvent(Node):
    """
    Runs REINVENT in a staged learning context.

    This node works by starting a REINVENT process with a special 'intercepting'
    external process to score the sampled SMILES. This interceptor simply accepts
    the SMILES on standard input and writes them to a location known by the node.
    The node then reads these SMILES and sends them to the output. The node then
    waits for scores to be received on the input, and writes them to a location
    known by the interceptor. The interceptor waits for the scores to be written
    and then reads them in to pass them to REINVENT on standard output. REINVENT
    can then perform its likelihood update and the cycle repeats until REINVENT
    exits or the maximum number of iterations is reached.

    """

    required_callables = ["reinvent"]
    """
    Requires REINVENT to be installed in a separate python environment
    and ideally be specified as an interpreter - script pair.

    """

    inp: Input[NDArray[np.float32]] = Input(optional=True)
    """Raw score input for the likelihood update"""

    out: Output[list[str]] = Output()
    """SMILES string output"""

    configuration: FileParameter[Annotated[Path, Suffix("toml", "json")]] = FileParameter()
    """ReInvent configuration file"""

    min_epoch: Parameter[int] = Parameter(default=5)
    """Minimum number of epochs to run"""

    max_epoch: Parameter[int] = Parameter(default=50)
    """Minimum number of epochs to run"""

    weight: Parameter[float] = Parameter(default=1.0)
    """Weight of the maize scoring component"""

    low: Parameter[float] = Parameter(default=0.0)
    """Low threshold for the sigmoid score transformation"""

    high: Parameter[float] = Parameter(default=1.0)
    """High threshold for the sigmoid score transformation"""

    k: Parameter[float] = Parameter(default=0.5)
    """Slope for the sigmoid score transformation"""

    reverse: Flag = Flag(default=False)
    """Whether to use a reverse sigmoid score transform"""

    batch_size: Parameter[int] = Parameter(default=128)
    """ReInvent batch size"""

    maize_backend: Flag = Flag(default=False)
    """Whether to use a special maize backend in Reinvent to enable weighted scores"""

    def _handle_smiles(self, worker: threading.Thread) -> None:
        self.logger.debug("Waiting for SMILES from Reinvent")
        while not TEMP_SMILES_FILE.exists():
            sleep(0.5)
            if not worker.is_alive():
                self.logger.debug("Reinvent has completed, exiting...")
                return
        with TEMP_SMILES_FILE.open("r") as file:
            smiles = [smi.strip() for smi in file.readlines()]
            self.logger.debug("Sending SMILES")
            self.out.send(smiles)
        TEMP_SMILES_FILE.unlink()

    def run(self) -> None:
        # Create the interceptor fake external process
        _write_interceptor(INTERCEPTOR_FILE, INTERCEPTOR)
        self.max_steps = self.max_epoch.value
        config, version = _patch_config(
            self.configuration.filepath,
            weight=self.weight.value,
            low=self.low.value,
            high=self.high.value,
            k=self.k.value,
            reverse=self.reverse.value,
            min_epochs=self.min_epoch.value,
            max_epochs=self.max_epoch.value,
            batch_size=self.batch_size.value,
            maize_backend=self.maize_backend.value,
        )

        if version >= 4:
            command = (
                f"{self.runnable['reinvent']} "
                f"--log-filename {REINVENT_LOGFILE.as_posix()} "
                f"-f {config.suffix.strip('.')} {config.as_posix()}"
            )

            # This allows us to keep track of the most recent REINVENT logs
            readlog = read_log(REINVENT_LOGFILE)
        else:
            command = f"{self.runnable['reinvent']} {config.as_posix()}"

        # Have to instantiate the executor in this process, as doing so in the subthread
        # will cause a python error due to the use of signals outside of the main thread.
        cmd = CommandRunner(working_dir=self.work_dir, rm_config=self.config.batch_config)
        threading.excepthook = _exception_handler

        # Start REINVENT in a separate thread (subprocess
        # starts a separate GIL-independent process)
        worker = threading.Thread(target=lambda: cmd.run(command))
        worker.start()
        self.logger.debug("Starting REINVENT worker with TID %s", worker.native_id)

        # Get the first set of SMILES
        self._handle_smiles(worker=worker)

        # Point to the tensorboard log directory if the user wants it
        tb_logs = list(self.work_dir.glob("tb_logs*"))
        if len(tb_logs) > 0:
            self.logger.info(
                "Tensorboard logs can be found at %s",
                tb_logs[-1].absolute().as_posix(),
            )

        epoch = 0
        for _ in self.loop():
            # Reinvent may terminate early
            self.logger.debug("Checking if Reinvent is still running")
            if not worker.is_alive():
                break

            if version >= 4:
                self.logger.info("ReInvent output: %s", readlog())

            self.logger.debug("Waiting for scores")
            scores = self.inp.receive()
            with TEMP_SCORES_FILE.open("w") as file:
                scores_data: dict[str, list[Any]] | list[Any]
                if scores.ndim == 2:
                    scores_data = {"scores": list(scores[0]), "weights": list(scores[1])}
                else:
                    scores_data = list(scores)
                self.logger.debug("Writing '%s'", scores_data)
                json.dump(scores_data, file)

            self._handle_smiles(worker=worker)
            self.logger.info("Sent new batch of SMILES, epoch %s", epoch)
            epoch += 1

        self.logger.info("Loop complete, stopping worker with TID %s", worker.native_id)
        worker.join(timeout=5)


class ReinventEntry(Node):
    """
    Specialized entrypoint for the REINVENT - Maize interface.

    Reads a JSON file containing generated SMILES and additional
    metadata, and outputs this information to be used in more
    generic workflows.

    Examples
    --------

    .. literalinclude:: ../../docs/reinvent-interface-example.yml
       :language: yaml
       :linenos:

    """

    data: FileParameter[Annotated[Path, Suffix("json")]] = FileParameter()
    """JSON input from Maize REINVENT scoring component"""

    out: Output[list[str]] = Output()
    """SMILES output"""

    out_metadata: Output[dict[str, Any]] = Output(optional=True)
    """Any additional metadata passed on by REINVENT"""

    def run(self) -> None:
        file = self.data.filepath
        with file.open() as inp:
            data = json.loads(inp.read())

        self.logger.debug("Received %s smiles", len(data["smiles"]))
        self.out.send(data["smiles"])
        self.out_metadata.send(data["metadata"])


class ReinventExit(Node):
    """
    Specialized exitpoint for the REINVENT - Maize interface.

    Creates a JSON file containing scores and relevances, which
    can be read in by the Maize scoring component in REINVENT.

    """

    inp: Input[list[IsomerCollection]] = Input()
    """Scored molecule input"""

    data: FileParameter[Annotated[Path, Suffix("json")]] = FileParameter(exist_required=False)
    """JSON output for Maize REINVENT scoring component"""

    def run(self) -> None:
        mols = self.inp.receive()
        self.logger.debug("Received %s mols", len(mols))

        # Add per-mol relevances if we have them
        relevance = [1.0 for _ in mols]
        if all(iso.has_tag("relevance") for mol in mols for iso in mol.molecules):
            relevance = [
                max(float(cast(float, iso.get_tag("relevance"))) for iso in mol.molecules)
                for mol in mols
            ]

        scores = [mol.best_score for mol in mols]
        self.logger.info("Sending %s scores: %s", len(mols), scores)
        data = dict(scores=scores, relevances=relevance)
        with self.data.filepath.open("w") as out:
            out.write(json.dumps(data))


@pytest.fixture
def reinvent_config(shared_datadir: Path) -> Path:
    return shared_datadir / "input-intercept.toml"


@pytest.fixture
def prior(shared_datadir: Path) -> Path:
    return shared_datadir / "random.prior.new"


@pytest.fixture
def agent(shared_datadir: Path) -> Path:
    return shared_datadir / "random.prior.new"


@pytest.fixture
def patch_config(prior: Path, agent: Path, reinvent_config: Path, tmp_path: Path) -> Path:
    with reinvent_config.open() as conf:
        data = toml.load(conf)
    data["parameters"]["prior_file"] = prior.absolute().as_posix()
    data["parameters"]["agent_file"] = agent.absolute().as_posix()
    new_config_file = tmp_path / "conf.toml"
    with new_config_file.open("w") as conf:
        toml.dump(data, conf)
    return new_config_file


def test_reinvent(temp_working_dir: Any, test_config: Any, patch_config: Any) -> None:
    n_epochs, n_batch = 5, 8
    scores = [np.random.rand(n_batch) for _ in range(n_epochs)]
    rig = TestRig(ReInvent, config=test_config)
    params = {
        "configuration": patch_config,
        "min_epoch": 3,
        "max_epoch": n_epochs,
        "batch_size": n_batch,
    }
    res = rig.setup_run(parameters=params, inputs={"inp": scores})
    data = res["out"].flush(timeout=0.5)
    assert 2 < len(data) <= n_epochs
    assert len(data[0]) == n_batch


@pytest.fixture
def reinvent_config_v3(shared_datadir: Path) -> Path:
    return shared_datadir / "input-intercept.json"


@pytest.fixture
def patch_config_v3(prior: Path, agent: Path, reinvent_config_v3: Path, tmp_path: Path) -> Path:
    with reinvent_config_v3.open() as conf:
        data = json.load(conf)
    data["parameters"]["reinforcement_learning"]["prior"] = prior.absolute().as_posix()
    data["parameters"]["reinforcement_learning"]["agent"] = agent.absolute().as_posix()
    new_config_file = tmp_path / "conf.json"
    with new_config_file.open("w") as conf:
        json.dump(data, conf)
    return new_config_file


def test_reinvent_v3(temp_working_dir: Any, test_config: Any, patch_config_v3: Any) -> None:
    n_epochs, n_batch = 5, 8
    scores = [np.random.rand(n_batch) for _ in range(n_epochs)]
    rig = TestRig(ReInvent, config=test_config)
    params = {
        "configuration": patch_config_v3,
        "min_epoch": 3,
        "max_epoch": n_epochs,
        "batch_size": n_batch,
    }
    res = rig.setup_run(parameters=params, inputs={"inp": scores})
    data = res["out"].flush(timeout=0.5)
    assert 2 < len(data) <= n_epochs
    assert len(data[0]) == n_batch


def test_reinvent_v3_weights(temp_working_dir: Any, test_config: Any, patch_config_v3: Any) -> None:
    n_epochs, n_batch = 5, 8
    scores = [np.random.rand(2, n_batch) for _ in range(n_epochs)]
    rig = TestRig(ReInvent, config=test_config)
    params = {
        "configuration": patch_config_v3,
        "min_epoch": 3,
        "max_epoch": n_epochs,
        "batch_size": n_batch,
        "maize_backend": True,
    }
    res = rig.setup_run(parameters=params, inputs={"inp": scores})
    data = res["out"].flush(timeout=0.5)
    assert 2 < len(data) <= n_epochs
    assert len(data[0]) == n_batch
