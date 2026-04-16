"""Schrodinger Ligprep prepares 3D small molecule conformers and isomers"""

# pylint: disable=import-outside-toplevel, import-error

import atexit
from collections.abc import Sequence
from enum import auto
import logging
from pathlib import Path
import shutil
from typing import Any
import psutil
import os
import shlex
from subprocess import CompletedProcess
from tempfile import mkdtemp
import time

from maize.core.node import Node
from maize.core.interface import Parameter, Flag
from maize.utilities.execution import CommandRunner, JobResourceConfig, RunningProcess, check_returncode, _simple_run, _log_command_output
from maize.utilities.utilities import StrEnum, unique_id
from maize.utilities.resources import cpu_count
from maize.utilities.validation import Validator


SCHRODINGER_LICENSE = "SCHROD_LICENSE_FILE"
SCHRODINGER_LOCAL = "SCHRODINGER_LOCALHOST_JOBSERVER_DIRECTORY"


log = logging.getLogger("run")


def has_license() -> bool:
    """``True`` if the system has a Schrodinger license, ``False`` otherwise."""
    return SCHRODINGER_LICENSE in os.environ


_BAD_SUB = ("No job server was found for -HOST",)


class _SchrodingerSubmissionStatus(StrEnum):
    SUCCESS = auto()
    HOST = auto()
    SERVER = auto()
    OTHER = auto()


def _query_schrodinger_submission(result: CompletedProcess[bytes]) -> _SchrodingerSubmissionStatus:
    """Queries the Schrodinger job submission status (this is before receiving a job ID)"""

    # This host can't run this command
    if result.returncode != 0 and any(
        fail in result.stdout.decode() + result.stderr.decode() for fail in _BAD_SUB
    ):
        return _SchrodingerSubmissionStatus.HOST

    # Some kind of error with the job server (e.g. GRPC
    # connection error 5, hard to track down and reproduce...)
    if result.returncode != 0 and any(
        fail in result.stdout.decode() + result.stderr.decode() for fail in _NO_SERVER
    ):
        return _SchrodingerSubmissionStatus.SERVER

    # Other error in command or input file
    if result.returncode != 0:
        return _SchrodingerSubmissionStatus.OTHER

    # Success
    return _SchrodingerSubmissionStatus.SUCCESS


_NO_SERVER = (
    "Error while dialing dial tcp",
    "No running local job server could be found",
    "Could not find a valid job server",
    "connection refused",
    "code = NotFound",
    "MMJOB_ERROR",
    "database disk image is malformed",
)


class _SchrodingerJobStatus(StrEnum):
    COMPLETED = auto()
    FAILED = auto()
    LICENSE = auto()
    RUNNING = auto()
    WAITING = auto()
    UNKNOWN = auto()
    SERVER = auto()
    STOPPED = auto()


def _update_result_log(result: CompletedProcess[bytes], job_name: str) -> CompletedProcess[bytes]:
    """Update the result with a Schrodinger logfile, if available"""
    file = Path(f"{job_name}.log")
    if file.exists():
        with file.open("rb") as log:
            result.stdout += f"\n--- {file.as_posix()} ---\n".encode()
            result.stdout += log.read()
    return result


def _kill_associated(*string: str) -> None:
    """Kills any process associated with a particular Schrodinger job ID"""
    for token in string:
        os.system(f"pkill -9 -f {token}")


def _kill_jobservers(maize_only: bool = True, server_id: str = "") -> list[int]:
    """Kills any currently running Schrodinger job server processes"""
    current_user = os.environ["USER"]
    killed: list[int] = []
    for proc in psutil.process_iter():
        if (
            # Schrodinger jobserver daemon
            proc.name().startswith("jobserverd")

            # Only our user
            and proc.username() == current_user

            # Only jobservers started from maize
            and ("maize-jobserver" in proc.cmdline()[0] or not maize_only)

            # Only a specific ID (given by maize)
            and server_id in proc.cmdline()[0]
        ):
            proc.kill()
            killed.append(proc.pid)
    return killed


def _job_server_running() -> bool:
    """Returns ``True`` if the Schrodinger job server is running"""
    res = _simple_run("jsc local-server-status")
    return "RUNNING" in res.stdout.decode() and res.returncode == 0


def _job_server_maize() -> bool:
    """Returns ``True`` if the Schrodinger job server was started by maize"""
    res = _simple_run("jsc local-server-dir")
    return "maize-jobserver" in res.stdout.decode() and res.returncode == 0


class Schrodinger(Node, register=False):
    FAILURES = (
        "Could not find a valid job server",
        "connection refused",
        "No running local job server could be found",
    )

    required_callables = ["jsc"]

    n_jobs: Parameter[int] = Parameter(default=1)
    """Number of jobs to spawn"""

    host: Parameter[str] = Parameter(default="localhost")
    """Host to use for job submission"""

    driverhost: Parameter[str] = Parameter(optional=True)
    """Host for the driver/coordinator process of distributed jobs.
    Overrides ``-HOST`` for the driver only, allowing the lightweight
    coordinator to run on a dedicated partition while subjobs use
    compute hosts specified by ``host`` (or ``subhost``).
    Only relevant when ``n_jobs > 1``. See the Schrodinger documentation
    section 'The HOST, DRIVERHOST, and SUBHOST Options' for details."""

    subhost: Parameter[str] = Parameter(optional=True)
    """Host(s) for subjobs of distributed jobs. Overrides ``-HOST``
    for subjobs only. When set, ``-HOST`` specifies only the driver
    location. See the Schrodinger documentation section 'Running
    Distributed Schrodinger Jobs' for details."""

    fallback: Flag = Flag(default=False)
    """If the host is not compatible, will fallback to 'localhost'"""

    job_server_temp: Flag = Flag(default=True)
    """Whether to run the Schrodinger job server in a dedicated temporary directory"""

    query_interval: Parameter[int] = Parameter(default=10)
    """
    The query interval for Schrodinger jobs. If you experience frequent failures and run
    large jobs you may want to increase this value to reduce the load on the job server.

    """
    _server_dir: Path | None = None
    _jobserver_handle: RunningProcess | None = None

    def _query_schrodinger_job(self, jobid: str) -> _SchrodingerJobStatus:
        """Queries the Schrodinger job server for a job"""
        res = _simple_run(f"{self.runnable['jsc']} info {jobid}")
        for line in res.stdout.decode().splitlines():
            if line.strip().startswith("Status:"):
                _, status = line.split()
                if status == "Failed" and any(
                    token in res.stdout.decode() for token in ("exit status 16", "exit status 1")
                ):
                    return _SchrodingerJobStatus.LICENSE
                return _SchrodingerJobStatus(status.upper())
            elif any(desc in (res.stdout.decode() + res.stderr.decode()) for desc in _NO_SERVER):
                return _SchrodingerJobStatus.SERVER
        return _SchrodingerJobStatus.UNKNOWN

    def _safe_log(self, *msg: Any, active: bool = True) -> None:
        """Debug logger, ensures safe logging even if the logger isn't registered yet"""
        if hasattr(self, "logger") and active:
            self.logger.debug(*msg)

    def _simple_run_verbose(self, command: str | list[str], log: bool = True) -> CompletedProcess[bytes]:
        """Run a local command and log the output"""
        res = _simple_run(command)
        if self.level in (logging.DEBUG, "debug"):
            self._safe_log(_log_command_output(res.stdout, res.stderr), active=log)
        return res

    def prepare(self) -> None:
        self._server_dir = Path(mkdtemp(prefix=f"maize-jobserver-{unique_id(32)}")).absolute()
        os.environ[SCHRODINGER_LOCAL] = self._server_dir.as_posix()
        self._safe_log(f"Setting {SCHRODINGER_LOCAL} to {self._server_dir.as_posix()}")

        self._restart_jobserver()
        atexit.register(self._cleanup_jobserver_temp, log=False)
        atexit.register(_kill_associated, self._server_dir.name)
        if self._jobserver_handle is not None:
            atexit.register(self._jobserver_handle.kill)

    def _shutdown(self) -> None:
        self._cleanup_jobserver_temp()
        if self._server_dir is not None:
            _kill_associated(self._server_dir.name)
        return super()._shutdown()

    def _download_result(self, jobid: str) -> CompletedProcess[bytes]:
        """Explicitly downloads any results from a job"""
        res = _simple_run(f"{self.runnable['jsc']} download {jobid}")
        if "no more files" in (stdout := res.stdout.decode()):
            self.logger.debug("Downloader failed:\n %s", stdout)
        return res

    def _cleanup_jobserver_temp(self, log: bool = True) -> None:
        """Attempts to kill the spawned jobserver and remove all residues"""
        self._safe_log("Stopping any running jobservers...", active=log)
        if self._jobserver_handle is not None:
            self._jobserver_handle.kill()
        self._safe_log("Killing all associated jobservers and remnants...", active=log)
        if self._server_dir is not None:
            killed = _kill_jobservers(server_id=self._server_dir.name)
            for pid in killed:
                self._safe_log("Killed PID %s", pid, active=log)

        if self._server_dir is not None and self._server_dir.exists():
            shutil.rmtree(
                self._server_dir,
                # Attempt at hardening against unreliable filesystems
                ignore_errors=True,
                onerror=lambda function, path, err: self.logger.warning("rmtree error: %s", err),
            )

    def _restart_jobserver(self) -> None:
        # self._cleanup_jobserver_temp()
        if self._server_dir is None:
            self._safe_log("Restart failed, server_dir not set...")
            return

        self._safe_log("Starting the job server...")
        self._simple_run_verbose(
            f"{self.runnable['jsc']} admin setup-server --dir {self._server_dir.as_posix()} "
            f"--host {os.environ['HOSTNAME']} --queue local --mode single-user "
            "--job-server-port 0 --file-store-port 0"
        )
        cmd = CommandRunner(name="jobserverd", raise_on_failure=False)
        self._jobserver_handle = cmd.run_async(f"{self._server_dir / 'bin' / 'jobserverd'} -dir {self._server_dir.absolute()}")

        # Waiting here is extremely important, as the starting command exits
        # immediately, but the server only becomes responsive later. Without
        # this we could end up in a deadlock with all Schrodinger commands
        # complaining about a missing server (that was just started).
        time.sleep(2)

    # If you think you can simplify this code, but failed, increment this counter:
    #
    #   failures = 2
    #
    # Schrodinger tools, and especially the forced job submission, have a large number of possible
    # failure modes. Here are just some I have encountered while attempting to interface with Glide:
    #
    # - licadmin STAT
    #   - No output
    #   - Incorrect output (underestimate of used licenses)
    #   - Missing license categories
    # - jsc
    #   - unknown job id
    #   - no running job server
    #   - communication failure with job server
    #   - files already downloaded
    # - commands (glide etc)
    #   - input error
    #   - no licenses available
    #   - no output generated
    #   - no connection:
    #       "transport: Error while dialing dial tcp [::1]:33369: connect: connection refused"
    #   - no connection: "GRPC connection error 5"
    #   - no job server: "No running local job server could be found"
    #   - segfault: "fatal error: unexpected signal during runtime execution [signal SIGSEGV:
    #                segmentation violation code=0x80 addr=0x0 pc=0x45ee60]"
    #   - no job server: "Error launching job: getJobRecord: Could not find a valid job server for
    #                     job d8426f52-77ec-11ee-8852-7cd30ac60bc4 among the following addresses:"
    #
    # Not all of these failures are explicitly accounted for,
    # instead certain modes are grouped and handled together.
    def _run_schrodinger_job(
        self,
        command: str | list[str],
        args: str | list[str] = "",
        working_dir: Path | None = None,
        verbose: bool = False,
        raise_on_failure: bool = True,
        name: str | None = None,
        validators: Sequence[Validator] | None = None,
        max_fail: int = -1,
        _n_failures: int = 0,
    ) -> CompletedProcess[bytes]:
        """
        Run a Schrodinger command.

        Schrodinger commands are special because they require communication
        with a job server that may be unreliable in some circumstances. We
        submit a command, get its ID, and then attempt to query the job server.

        Parameters
        ----------
        command
            Base command without arguments to run as a single string, or a list of strings
        args
            The positional arguments to run the command on
        working_dir
            Optional working directory
        verbose
            If ``True`` will also log any STDOUT or STDERR output
        raise_on_failure
            Whether to raise an exception on failure, or whether to just return `False`.
        name
            Name of the job, will also determine the names of the output files
        validators
            One or more `Validator` instances that will
            be called on the result of the command.
        max_fail
            Maximum number of job running failures, if ``-1`` will try infinitely often

        Returns
        -------
        subprocess.CompletedProcess[bytes]
            Result of the execution, including STDOUT and STDERR

        Raises
        ------
        ProcessError
            If the returncode was not zero

        """
        if max_fail >= 0 and _n_failures >= max_fail:
            return CompletedProcess(args=command, returncode=1)

        if isinstance(command, str):
            command = shlex.split(command)
        name = name or unique_id(12)

        # Only add the extras once (first try) as the command
        # will be correctly setup all subsequent attempts
        if _n_failures == 0:
            # Cleanup command from previously specified values
            for token in ("-JOBNAME", "-HOST", "-NJOBS", "-DRIVERHOST", "-SUBHOST"):
                if token in command:
                    idx = command.index(token)
                    command.pop(idx + 1)
                    command.remove(token)

            # Provide a unique name to avoid overwrite prompts
            # and allow easier concatenation of output
            command.extend(["-JOBNAME", name])

            # Set correct number of jobs depending on host
            host = self.host.value
            if host == "localhost":
                host = f"{host}:{self.n_jobs.value}"
            command.extend(["-HOST", host])
            command.extend(["-NJOBS", str(self.n_jobs.value)])

            # For distributed jobs (n_jobs > 1), allow separate routing of the
            # driver/coordinator process and subjobs. This follows Schrodinger's
            # Job Control conventions: -DRIVERHOST overrides -HOST for the driver,
            # -SUBHOST overrides -HOST for subjobs. When neither is set, all
            # processes use the -HOST value (existing behavior).
            if self.driverhost.is_set:
                command.extend(["-DRIVERHOST", self.driverhost.value])
            if self.subhost.is_set:
                command.extend(["-SUBHOST", self.subhost.value])

            # Add the actual args at the end to maintain correct ordering
            if isinstance(args, str):
                args = shlex.split(args)
            command.extend(args)

        def _restart(n_fail: int) -> CompletedProcess[bytes]:
            return self._run_schrodinger_job(
                command,
                args,
                working_dir=working_dir,
                verbose=verbose,
                raise_on_failure=raise_on_failure,
                name=name,
                validators=validators,
                max_fail=max_fail,
                _n_failures=n_fail,
            )

        # Submit job and check for errors upon submission
        self.logger.debug("Running Schrodinger job '%s'", " ".join(command))
        cmd = CommandRunner(raise_on_failure=False, working_dir=working_dir, validators=validators)
        result = cmd.run_only(command, verbose=verbose)

        match _query_schrodinger_submission(result):
            case _SchrodingerSubmissionStatus.SERVER:
                self.logger.warning(
                    "Job server communication failure, restarting jobserver\n %s",
                    result.stdout.decode(),
                )
                self._restart_jobserver()
                time.sleep(5)
                return _restart(n_fail=_n_failures + 1)

            case _SchrodingerSubmissionStatus.HOST:
                self.logger.warning(
                    "Host %s is unavailable, you may need to run 'jsc cert get <host>:<port>'",
                    self.host.value,
                )
                if self.fallback.value and self.host.value != "localhost":
                    n_jobs = min(self.n_jobs.value, cpu_count())
                    self.host.set("localhost")
                    self.logger.warning(
                        "Falling back to 'localhost', using %s jobs (instead of %s jobs)",
                        n_jobs,
                        self.n_jobs.value,
                    )
                    self.n_jobs.set(n_jobs)
                    return _restart(n_fail=0)
                check_returncode(result, raise_on_failure=raise_on_failure, logger=self.logger)
                self._cleanup_jobserver_temp()
                return result

            case _SchrodingerSubmissionStatus.OTHER:
                self.logger.debug("Other failure, returncode %s", result.returncode)
                check_returncode(result, raise_on_failure=raise_on_failure, logger=self.logger)
                self._cleanup_jobserver_temp()
                return result

            case _SchrodingerSubmissionStatus.SUCCESS:
                self.logger.debug("Submission succeeded, now monitoring...")

        # Submission success, get the ID and monitor progression
        n_fails = 0
        _, jobid = result.stdout.decode().split()
        self.logger.debug("Monitoring job with Schrodinger ID %s", jobid)
        while not self.signal.is_set():
            time.sleep(self.query_interval.value)
            match self._query_schrodinger_job(jobid):
                # Job done
                case _SchrodingerJobStatus.COMPLETED | _SchrodingerJobStatus.FAILED:
                    self.logger.debug("Job done, returncode: %s", result.returncode)
                    self._download_result(jobid)
                    result = _update_result_log(result, job_name=name)
                    if result.returncode != 0:
                        self.logger.warning(
                            "Schrodinger job failed, run 'jsc postmortem %s' for more information",
                            jobid,
                        )
                    check_returncode(result, raise_on_failure=raise_on_failure, logger=self.logger)
                    cmd.validate(result)
                    self._cleanup_jobserver_temp()
                    return result

                # Still running (not 100% reliable, may be waiting for licenses in some situations)
                case _SchrodingerJobStatus.RUNNING:
                    self.logger.debug("Probably running, potentially still waiting for licenses...")

                # Waiting for licenses
                case _SchrodingerJobStatus.WAITING:
                    self.logger.debug("Waiting for licenses...")

                # Timed out due to not enough licenses
                case _SchrodingerJobStatus.LICENSE:
                    self.logger.debug("Timed out due to licenses, retrying...")
                    return _restart(n_fail=_n_failures + 1)

                # Job server died or never started properly
                case _SchrodingerJobStatus.SERVER | _SchrodingerJobStatus.STOPPED:
                    self.logger.warning("Job server communication failure, restarting server")
                    self._restart_jobserver()

                # Hopefully temporary communication failure
                case _SchrodingerJobStatus.UNKNOWN:
                    # After 5 comms failures we assume something went wrong and resubmit the job
                    if n_fails == 5:
                        self.logger.warning("Multiple unknown errors querying job, resubmitting")
                        _kill_associated(jobid, name)
                        return _restart(n_fail=_n_failures + 1)

                    if n_fails == 0:
                        self.logger.warning("Unknown job querying error")

                    n_fails += 1

        self._cleanup_jobserver_temp()
        return result

    # Always make sure we're running the preliminary commands
    def run_command(
        self,
        command: str | list[str],
        working_dir: Path | None = None,
        validators: Sequence[Validator] | None = None,
        verbose: bool = False,
        raise_on_failure: bool = True,
        command_input: str | None = None,
        pre_execution: str | list[str] | None = None,
        batch_options: JobResourceConfig | None = None,
        prefer_batch: bool = False,
        timeout: float | None = None,
        cuda_mps: bool = False,
    ) -> CompletedProcess[bytes]:
        # While we will generally use a token guard with Schrodinger tools,
        # there are rare situations where we might expect to have licenses
        # available, but because of a short lag (< 5s) another user might
        # have claimed them in this short window. In this case, Schrodinger
        # will wait for 30s three times to acquire the tokens and if not
        # successful, fail with exit code 16. In this situation we retry the
        # command, otherwise this is handled like any other use of run_command.
        while not self.signal.is_set():
            ret = super().run_command(
                command=command,
                working_dir=working_dir,
                validators=validators,
                verbose=verbose,
                raise_on_failure=False,
                command_input=command_input,
                pre_execution=pre_execution,
                batch_options=batch_options,
                prefer_batch=prefer_batch,
                timeout=timeout,
                cuda_mps=cuda_mps,
            )
            if ret.returncode != 16:
                check_returncode(ret, raise_on_failure=raise_on_failure, logger=self.logger)
                return ret
            self.logger.warning("Command failed due to unavailable tokens, trying again...")

        # This is a fallback incase we exit the workflow while waiting for licenses
        return CompletedProcess(command, returncode=1)
