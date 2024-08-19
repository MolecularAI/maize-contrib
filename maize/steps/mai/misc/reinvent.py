"""Interface to REINVENT"""

from collections.abc import Callable
import json
import os
from pathlib import Path
import stat
import sys
from time import sleep
from typing import TYPE_CHECKING, Annotated, Any, cast
import pytest

import toml
import numpy as np
from numpy.typing import NDArray
from string import Template


from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors


from maize.core.node import Node
from maize.core.workflow import Workflow
from maize.core.interface import Input, Output, Parameter, FileParameter, Suffix, Flag
from maize.steps.io import LoadData, Return
from maize.utilities.chem import IsomerCollection
from maize.utilities.execution import CommandRunner, RunningProcess
from maize.utilities.testing import TestRig
from maize.utilities.io import setup_workflow, Config
from maize.utilities.validation import ContentValidator, FileValidator


if TYPE_CHECKING:
    from maize.core.graph import Graph


REINVENT_LOGFILE = Path("reinvent.log")
DEFAULT_PATCHED_CONFIG = Path("config.toml")
DEFAULT_SCORING_CONFIG = Path("scoring.toml")
DEFAULT_SOURCE_CONFIG = Path("source.smi")
DEFAULT_WF_CONFIG = Path("maize_rnv_wf.yml")

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
    prior: Path | None = None,
    agent: Path | None = None,
    maize_backend: bool = False,
) -> Path:
    """Patch the REINVENT config to allow interception of SMILES."""
    score_conf = {
        "name": "maize",
        "weight": weight,
        "params": {
            "executable": "./intercept.py",
            "args": "",
        },
        "transform": {
            "low": low,
            "high": high,
            "k": k,
            "type": "reverse_sigmoid" if reverse else "sigmoid",
        },
    }
    with path.open() as file:
        if path.suffix == ".json":
            conf = json.load(file)
        elif path.suffix == ".toml":
            conf = toml.load(file)
        else:
            raise IOError(f"Unable to read REINVENT config '{path.as_posix()}'")

    conf["stage"][0]["scoring"]["component"].append({"ExternalProcess": {"endpoint": [score_conf]}})
    conf["stage"][0]["min_steps"] = min_epochs
    conf["stage"][0]["max_steps"] = max_epochs
    conf["parameters"]["batch_size"] = batch_size

    if prior is not None:
        conf["parameters"]["prior_file"] = prior.absolute().as_posix()
    if agent is not None:
        conf["parameters"]["agent_file"] = agent.absolute().as_posix()

    patched_file = DEFAULT_PATCHED_CONFIG.with_suffix(path.suffix)
    with patched_file.open("w") as out:
        if path.suffix == ".json":
            json.dump(conf, out)
        elif path.suffix == ".toml":
            toml.dump(conf, out)
    return patched_file


class read_log:
    def __init__(self, logfile: Path) -> None:
        self.logfile = logfile
        self._new_lines = 0

    def __call__(self) -> str:
        """Read the ReInvent logfile and format"""
        with self.logfile.open() as log:
            lines = log.readlines()
            all_lines = len(lines)
            
            # Avoid printing everything if nothing has changed
            if all_lines == self._new_lines:
                return ""

            msg = "\n"
            msg += "---------------- STDOUT ----------------\n"
            msg += "".join(lines[-(all_lines - self._new_lines) :]) + "\n"
            msg += "---------------- STDOUT ----------------\n"
            self._new_lines = all_lines
        return msg


class Mol2MolStandalone(Node):
    """
    Runs REINVENT/Mol2Mol staged learning end-to-end as single node with predefined config.

    This node does not allow dynamic interaction with REINVENT nor the use of other nodes in
    the graph for scoring (see the standard ReInvent node for that). This is a "black box"
    version that takes a source molcule and scoring configuration file as input  and returns
    a path to the job results, allowing simple near neighor expansion / molecule generation
    tasks to be embedded into graphs.

    """

    required_callables = ["reinvent"]
    """
    Requires REINVENT to be installed in a separate python environment
    and ideally be specified as an interpreter - script pair.

    """

    input_source: Input[IsomerCollection] = Input()
    """A source molecule to start ideation"""

    input_runconfig: Input[Annotated[Path, Suffix("toml")]] = Input()
    """
    A standalone boilerplate config to used in staged learning
    with mol2mol, must reference "reference_compound.smi" as input

    """

    input_scoring: Input[Annotated[Path, Suffix("toml")]] = Input()
    """
    A standalone scoring config to be used in staged learning.

    This needs to be a template file and will expand keys in a reference reinvent input:
    
    ===============  ================  ================
    Feature          Low               High
    ===============  ================  ================
    molar weight     ``$mw_low``       ``$mw_high``
    clogp            ``$clogp_low``    ``$clogp_low``
    HB donors        ``$hbd_low``      ``$hbd_high``
    # halogens       ``$halo_low``     ``$halo_high``
    # heteroatoms    ``$hetero_low``   ``$hetero_high``
    docking score    ``$docking_low``  ``$docking_high (optional)``
    ===============  ================  ================

    """

    docking_reference_score: Input[float] = Input(optional=True)
    """Should docking be used? It must be referenced in input_scoring"""

    docking_reference_workflow: Input[Annotated[Path, Suffix("yml")]] = Input(optional=True)
    """
    A yml maize workflow template which must contain
    ``$grid`` and optionally ``$referenceligand``

    """

    docking_grid: Input[Annotated[Path, Suffix("zip")]] = Input(optional=True)
    """This is a the input docking grid"""

    docking_reference_ligand: Input[Annotated[Path, Suffix("mae", "sdf")]] = Input(optional=True)
    """Should docking be used? This is a reference ligand"""

    out: Output[Annotated[Path, Suffix("csv")]] = Output()
    """Path to results file from staged learning"""

    max_steps: Parameter[int] = Parameter(default=50)
    """Minimum number of epochs to run"""

    batch_size: Parameter[int] = Parameter(default=128)
    """ReInvent batch size"""

    scoring_window_overwrite: Parameter[dict[str, float]] = Parameter(default={})
    """Overwrite ranges to expand the Mol2MolInput"""

    reinvent_dotenv: FileParameter[Path] = FileParameter(optional=True)
    """Optional path to ReInvent dotenv file for setting ReInvent variables"""

    def run(self) -> None:
        boilerplate_path = self.input_runconfig.receive()
        scoring_path = self.input_scoring.receive()

        # template config values
        boilerplate_config = toml.load(open(boilerplate_path.as_posix(), "r"))
        boilerplate_config["parameters"]["batch_size"] = self.batch_size.value
        boilerplate_config["parameters"]["smiles_file"] = DEFAULT_SOURCE_CONFIG.as_posix()
        boilerplate_config["stage"][0]["max_steps"] = self.max_steps.value
        boilerplate_config["stage"][0]["scoring"]["filename"] = DEFAULT_SCORING_CONFIG.as_posix()

        # set up scoring
        source_mol = self.input_source.receive().molecules[0]

        ## compute core descriptors
        smiles = source_mol.to_smiles()
        mw = round(Descriptors.MolWt(source_mol._molecule), 0)
        clogp = round(Crippen.MolLogP(source_mol._molecule), 1)
        hbd = rdMolDescriptors.CalcNumHBD(source_mol._molecule)
        halos = len(source_mol._molecule.GetSubstructMatches(Chem.MolFromSmarts("[F,Cl]")))
        heteros = len(source_mol._molecule.GetSubstructMatches(Chem.MolFromSmarts("[!#6]")))

        # compute updated scores
        score_ranges = {
            "mw_up": 75,
            "mw_down": 75,
            "clogp_up": 1.5,
            "clogp_down": 1.5,
            "hbd_up": 2,
            "hbd_down": 0,
            "halo_up": 4,
            "halo_down": 0,
            "hetero_up": 0,
            "hetero_down": 6,
            "docking_up": 7,
            "docking_down": 1,
        }

        if self.scoring_window_overwrite.value:
            score_ranges.update(self.scoring_window_overwrite.value)

        scoring_params = {
            "mw_low": mw - score_ranges["mw_down"],
            "mw_high": mw + score_ranges["mw_up"],
            "clogp_low": round(clogp - score_ranges["clogp_down"], 2),
            "clogp_high": round(clogp + score_ranges["clogp_up"], 2),
            "hbd_low": hbd - score_ranges["hbd_down"],
            "hbd_high": hbd + score_ranges["hbd_up"],
            "halo_low": halos - score_ranges["halo_down"],
            "halo_high": halos + score_ranges["halo_up"],
            "hetero_low": heteros - score_ranges["hetero_down"],
            "hetero_high": heteros + score_ranges["hetero_up"],
        }

        # check if should do docking
        docking_reference_score = self.docking_reference_score.receive_optional()
        docking_reference_workflow_file = self.docking_reference_workflow.receive_optional()
        docking_grid_path = self.docking_grid.receive_optional()
        docking_reference_ligand_path = self.docking_reference_ligand.receive_optional()

        if docking_reference_score:
            self.logger.debug("Using docking in Mol2Mol standalone")

            # check input is consistent
            if not docking_reference_workflow_file or not docking_grid_path:
                raise ValueError(
                    "Insufficient inputs for docking, grid and reference workflow needed"
                )

            # set params for docking wf
            docking_params = {"grid": docking_grid_path}
            if docking_reference_ligand_path:
                docking_params["referenceligand"] = docking_reference_ligand_path

            docking_wf_template = Template(
                open(docking_reference_workflow_file.as_posix(), "r").read()
            )
            docking_wf = docking_wf_template.safe_substitute(docking_params)

            wf_file = self.work_dir / DEFAULT_WF_CONFIG
            with wf_file.open("w") as out:
                out.write(docking_wf)

            # set scoring conf
            scoring_params.update(
                {
                    "docking_low": round(docking_reference_score - score_ranges["docking_down"], 2),
                    "docking_high": round(docking_reference_score + score_ranges["docking_up"], 2),
                }
            )

        # check input is consistent
        elif docking_reference_workflow_file or docking_grid_path or docking_reference_ligand_path:
            raise ValueError("docking_reference_score needed for for docking in Mol2Mol standalone")

        # now we can set the scoring conf
        score_config_template = Template(open(scoring_path.as_posix(), "r").read())
        score_config_str = score_config_template.safe_substitute(scoring_params)
        score_config = toml.loads(score_config_str)

        if docking_reference_score:
            for score_comp in score_config["component"]:
                if "Maize" in score_comp.keys():
                    score_comp["Maize"]["endpoint"][0]["params"]["workflow"] = wf_file.as_posix()

        # write needed files
        config_file = self.work_dir / DEFAULT_PATCHED_CONFIG
        with config_file.open("w") as out:
            toml.dump(boilerplate_config, out)

        scoring_file = self.work_dir / DEFAULT_SCORING_CONFIG
        with scoring_file.open("w") as out:
            toml.dump(score_config, out)

        source_file = self.work_dir / DEFAULT_SOURCE_CONFIG
        with source_file.open("w") as out:
            out.write(smiles)

        # target output paths
        target_output_path = self.work_dir / "summary_1.csv"
        target_log_path = self.work_dir / REINVENT_LOGFILE

        # build the command
        command = (
            f"{self.runnable['reinvent']} "
            f"--log-filename {REINVENT_LOGFILE.as_posix()} "
            f"-f {config_file.suffix.strip('.')} {config_file.as_posix()}"
        )

        # set env variables if needed
        if self.reinvent_dotenv.is_set:
            command = command + f" --dotenv-filename {self.reinvent_dotenv.filepath}"

        # This allows us to keep track of the most recent REINVENT logs
        readlog = read_log(target_log_path)

        # run the command

        cmd = CommandRunner(
            working_dir=self.work_dir,
            rm_config=self.config.batch_config,
            validators=[
                FileValidator(target_output_path),
                ContentValidator({target_log_path: ["Finished LREINVENT"]}),
            ],
        )

        worker = cmd.run_async(command)
        self.logger.debug("Starting REINVENT worker")

        sleep(120)
        self.logger.debug("wait complete, checking for logs")

        # Point to the tensorboard log directory if the user wants it
        tb_logs = list(self.work_dir.glob("tb_logs*"))
        if len(tb_logs) > 0:
            self.logger.info(
                "Tensorboard logs can be found at %s",
                tb_logs[-1].absolute().as_posix(),
            )

        count = 0
        while worker.is_alive() and not self.signal.is_set():
            self.logger.debug(f"Checking if ReInvent is still running, counter: {count}")
            if Path.is_file(target_output_path):
                self.logger.info("ReInvent output: %s", readlog())
            else:
                self.logger.info("ReInvent output not yet available")
            count += 1
            sleep(60)

        self.logger.info("Loop complete, stopping worker")
        worker.kill(timeout=5)
        worker.wait()
        self.out.send(target_output_path)


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

    prior: FileParameter[Path] = FileParameter(optional=True)
    """ReInvent prior file"""

    agent: FileParameter[Path] = FileParameter(optional=True)
    """ReInvent agent file"""

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

    reinvent_dotenv: FileParameter[Path] = FileParameter(optional=True)
    """Optional path to ReInvent dotenv file for setting ReInvent variables"""

    def _handle_smiles(self, worker: RunningProcess) -> None:
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

        prior = self.prior.filepath if self.prior.is_set else None
        agent = self.agent.filepath if self.agent.is_set else None

        config = _patch_config(
            self.configuration.filepath,
            weight=self.weight.value,
            low=self.low.value,
            high=self.high.value,
            k=self.k.value,
            reverse=self.reverse.value,
            min_epochs=self.min_epoch.value,
            max_epochs=self.max_epoch.value,
            batch_size=self.batch_size.value,
            prior=prior,
            agent=agent,
            maize_backend=self.maize_backend.value,
        )

        command = (
            f"{self.runnable['reinvent']} "
            f"--log-filename {REINVENT_LOGFILE.as_posix()} "
            f"-f {config.suffix.strip('.')} {config.as_posix()}"
        )

        # set env variables if needed
        if self.reinvent_dotenv.is_set:
            command = command + f" --dotenv-filename {self.reinvent_dotenv.filepath}"

        # This allows us to keep track of the most recent REINVENT logs
        readlog = read_log(REINVENT_LOGFILE)

        cmd = CommandRunner(working_dir=self.work_dir, rm_config=self.config.batch_config)
        worker = cmd.run_async(command)
        self.logger.debug("Starting REINVENT worker")

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
        for _ in range(self.max_epoch.value):
            # Reinvent may terminate early
            self.logger.debug("Checking if ReInvent is still running")
            if not worker.is_alive() or self.signal.is_set():
                break

            self.logger.info("ReInvent output: %s", readlog())

            self.logger.debug("Waiting for scores")
            scores = self.inp.receive()
            with TEMP_SCORES_FILE.open("w") as file:
                scores_data = {"version": 4, "payload": {"predictions": list(scores)}}
                self.logger.debug("Writing '%s'", scores_data)
                json.dump(scores_data, file)

            self._handle_smiles(worker=worker)
            self.logger.info("Sent new batch of SMILES, epoch %s", epoch)
            epoch += 1

        self.logger.info("Loop complete, stopping worker")
        worker.kill(timeout=5)
        worker.wait()


class StripEpoch(Node):
    """Isolates the epoch from REINVENT metadata"""

    inp: Input[dict[str, Any]] = Input()
    """Metadata dictionary input"""

    out: Output[int] = Output()
    """Epoch index output"""

    def run(self) -> None:
        data = self.inp.receive()
        self.out.send(data["iteration"])


class ReinventEntry(Node):
    """
    Specialized entrypoint for the REINVENT - Maize interface.

    Reads a JSON file containing generated SMILES and additional
    metadata, and outputs this information to be used in more
    generic workflows.

    Examples
    --------

    .. literalinclude:: ../../reinvent-interface-example.yml
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

        msg = "Received the following data from REINVENT\n"
        for smi in data["smiles"]:
            msg += f"    {smi:<75}\n"
        self.logger.debug(msg)

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

    tag: Parameter[str] = Parameter(optional=True)
    """The score name to send, will use the primary score if not given"""

    def run(self) -> None:
        mols = self.inp.receive()
        self.logger.debug("Received %s mols", len(mols))

        names = [mol.name for mol in mols]

        # Add per-mol relevances if we have them
        relevance = [1.0 for _ in mols]
        if all(iso.has_tag("relevance") for mol in mols for iso in mol.molecules):
            relevance = [
                max(float(cast(float, iso.get_tag("relevance"))) for iso in mol.molecules)
                for mol in mols
            ]

        if self.tag.is_set:
            key = self.tag.value
            scores = []
            for mol in mols:
                scores.append(mol.scores[key] if key in mol.scores else np.nan)
        else:
            scores = [float(mol.primary_score) for mol in mols]

        msg = "Sending the following data back to REINVENT\n"
        for mol in mols:
            msg += f"    {mol.smiles:<75}  {mol.primary_score:5.4f}\n"
        self.logger.debug(msg)

        self.logger.info("Sending %s scores: %s", len(mols), scores)
        data = dict(scores=scores, relevances=relevance, names=names)
        with self.data.filepath.open("w") as out:
            out.write(json.dumps(data))


@pytest.fixture
def reinvent_config(shared_datadir: Path) -> Path:
    return shared_datadir / "input-intercept.toml"


@pytest.fixture
def prior(shared_datadir: Path) -> Path:
    return shared_datadir / "reinvent.prior"


@pytest.fixture
def agent(shared_datadir: Path) -> Path:
    return shared_datadir / "reinvent.prior"


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


@pytest.mark.needs_node("reinvent")
def test_reinvent(temp_working_dir: Path, test_config: Config, patch_config: Path) -> None:
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
    data = res["out"].flush(timeout=20)
    assert len(data) == n_epochs
    assert 1 < len(data[0]) <= n_batch
