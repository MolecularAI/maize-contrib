"""Schrodinger Ligprep prepares 3D small molecule conformers and isomers"""

# pylint: disable=import-outside-toplevel, import-error

from collections.abc import Sequence
from enum import auto
import functools
import logging
from pathlib import Path
import shutil
import psutil
import os
import re
import shlex
from subprocess import CompletedProcess
from tempfile import mkdtemp
import time

from maize.core.node import Node
from maize.core.interface import Parameter, Flag
from maize.utilities.execution import CommandRunner, JobResourceConfig, check_returncode
from maize.utilities.utilities import StrEnum, unique_id
from maize.utilities.resources import cpu_count
from maize.utilities.validation import Validator


SCHRODINGER_LICENSE = "SCHROD_LICENSE_FILE"
SCHRODINGER_LOCAL = "SCHRODINGER_LOCALHOST_JOBSERVER_DIRECTORY"


log = logging.getLogger("run")


def has_license() -> bool:
    """``True`` if the system has a Schrodinger license, ``False`` otherwise."""
    return SCHRODINGER_LICENSE in os.environ


class TokenGuard:
    """
    Schrodinger token handling. Checks the number of available tokens and waits for availability.

    Parameters
    ----------
    pre
        Command to run prior to running ``licadmin``

    """

    LICENSE_RE = re.compile(
        r"Users\s+of\s+(.*):.*of\s+(\d+)\s+licenses\s+issued;.*of\s+(\d+)\s+licenses\s+in\s+use"
    )

    def __init__(self, parent: Node, pre: str | list[str], max_retries: int = 3) -> None:
        self.pre_execution = pre
        self.parent = parent
        self.max_retries = max_retries
        self._tokens: dict[str, tuple[int, int]] = {}
        if not has_license():
            raise EnvironmentError(
                f"No valid Schrodinger license found "
                f"({SCHRODINGER_LICENSE}={os.environ.get(SCHRODINGER_LICENSE)})"
            )
        self.query()

    @property
    def licenses(self) -> dict[str, int]:
        """Provides the available software and associated number of licenses"""
        return {key: val for key, (val, _) in self._tokens.items()}

    def query(self, _retry: int = 0) -> None:
        """Queries the license server for available tokens"""
        out = CommandRunner(raise_on_failure=False).run(
            "licadmin STAT", pre_execution=self.pre_execution
        )
        # licadmin STAT can sometimes fail with:
        # Error getting status: Cannot read data from license server system. (-16,287)
        if out.returncode != 0:
            if _retry < self.max_retries:
                time.sleep(2)
                self.query(_retry=_retry + 1)
            return

        tokens = {}
        for line in out.stdout.decode().splitlines():
            if match := re.match(self.LICENSE_RE, line):
                name, total, used = match.groups()
                tokens[name] = (int(total), int(used))

        self._tokens.update(tokens)

    def check(self, key: str, number: int) -> bool:
        """
        Checks if licenses are available for software.

        Parameters
        ----------
        key
            Name of the software to check
        number
            Number of licenses to check for

        Returns
        -------
        bool
            ``True`` if licenses are available or the name is
            not in the available licenses, ``False`` otherwise

        """
        if key not in self._tokens:
            log.warning(
                "Could not find '%s' under available Schrodinger "
                "licenses, tentatively proceeding without check...",
                key,
            )
            return True
        total, used = self._tokens[key]
        return (used + number) <= total

    def wait(self, key: str, number: int = 1, timeout: float | None = None) -> None:
        """
        Wait for a specific number of licenses to become available.

        Parameters
        ----------
        key
            Name of the required software
        number
            Number of licenses to request
        timeout
            Timeout in seconds, or ``None`` to wait indefinitely

        Raises
        ------
        TimeoutError
            If the required number of licenses could not be found in time

        """
        if self.check(key, number):
            return

        start = time.time()
        while not self.parent.signal.is_set():
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(f"Could not request {number} tokens for '{key}'")

            time.sleep(5)
            self.query()
            if self.check(key, number):
                break


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


def _query_schrodinger_job(jobid: str) -> _SchrodingerJobStatus:
    """Queries the Schrodinger job server for a job"""
    cmd = CommandRunner(raise_on_failure=False)
    res = cmd.run_only(f"jsc info {jobid}")
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


def _kill_jobservers(maize_only: bool = True) -> list[int]:
    """Kills any currently running Schrodinger job server processes"""
    current_user = os.environ["USER"]
    killed = []
    for proc in psutil.process_iter():
        if (
            proc.name().startswith("jobserverd")
            and proc.username() == current_user
            and ("maize-jobserver" in proc.cmdline[0] or not maize_only)
        ):
            proc.kill()
            killed.append(proc.pid)
    return killed


def _job_server_running() -> bool:
    """Returns ``True`` if the Schrodinger job server is running"""
    res = CommandRunner(raise_on_failure=False).run_only("jsc local-server-status")
    return "RUNNING" in res.stdout.decode() and res.returncode == 0


def _job_server_maize() -> bool:
    """Returns ``True`` if the Schrodinger job server was started by maize"""
    res = CommandRunner(raise_on_failure=False).run_only("jsc local-server-dir")
    return "maize-jobserver" in res.stdout.decode() and res.returncode == 0


def _show_all_ports() -> str:
    res = CommandRunner(raise_on_failure=False).run_only("netstat -tulnp")
    return "\n".join(line for line in res.stdout.decode().splitlines() if "jobserverd" in line)


class Schrodinger(Node, register=False):
    FAILURES = (
        "Could not find a valid job server",
        "connection refused",
        "No running local job server could be found",
    )

    guard: TokenGuard

    n_jobs: Parameter[int] = Parameter(default=1)
    """Number of jobs to spawn"""

    host: Parameter[str] = Parameter(default="localhost")
    """Host to use for job submission"""

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

    def _prepare(self) -> None:
        # We are overriding _prepare here to be able to set the SCHRODINGER_LOCALHOST_JOBSERVER_DIRECTORY
        # environment variable *before* loading the Schrodinger module, as the latter will automatically
        # start a non-local jobserver in a directory not explicitly set by us. The suspicion here is that
        # this non-local server is interferring with any local dedicated ones.
        if self.job_server_temp.value:
            self._server_dir = Path(mkdtemp(prefix=f"maize-jobserver-{unique_id(32)}")).absolute()
            os.environ[SCHRODINGER_LOCAL] = self._server_dir.as_posix()
            if hasattr(self, "logger"):
                self.logger.debug(f"Setting {SCHRODINGER_LOCAL} to {self._server_dir.as_posix()}")
        return super()._prepare()

    def prepare(self) -> None:
        if not _job_server_maize() or not _job_server_running():
            self._restart_jobserver()

    def _download_result(self, jobid: str) -> CompletedProcess[bytes]:
        """Explicitly downloads any results from a job"""
        cmd = CommandRunner(raise_on_failure=False)
        res = cmd.run_only(f"jsc download {jobid}")
        if "no more files" in (stdout := res.stdout.decode()):
            self.logger.debug("Downloader failed:\n %s", stdout)
        return res

    def _cleanup_jobserver_temp(self) -> None:
        if self._server_dir is not None:
            shutil.rmtree(self._server_dir)

    def _restart_jobserver(self) -> CompletedProcess[bytes]:
        cmd = CommandRunner(raise_on_failure=False)
        self.logger.debug("Stopping any running jobservers...")
        cmd.run_only("jsc local-server-stop", verbose=True)
        self._cleanup_jobserver_temp()
        if self.job_server_temp.value and self._server_dir is not None:
            # It is very important that this directory is globally accessible
            # so that all Schrodinger nodes can use the same jobserver
            self.logger.debug(f"Setting server directory to {self._server_dir.as_posix()}")
            cmd.run_only(f"jsc local-server-dir --set {self._server_dir.as_posix()}", verbose=True)

        self.logger.debug("Starting the job server...")
        res = cmd.run_only("jsc local-server-start", verbose=True)

        # Waiting here is extremely important, as the starting command exits
        # immediately, but the server only becomes responsive later. Without
        # this we could end up in a deadlock with all Schrodinger commands
        # complaining about a missing server (that was just started).
        time.sleep(5)
        self.logger.debug("Restart %ssuccessful", "" if _job_server_running() else "not ")
        self.logger.debug("Ports in use by jobservers:\n %s", _show_all_ports())
        return res


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
            for token in ("-JOBNAME", "-HOST", "-NJOBS"):
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

            # Add the actual args at the end to maintain correct ordering
            if isinstance(args, str):
                args = shlex.split(args)
            command.extend(args)

        _restart = functools.partial(
            lambda n_fail: self._run_schrodinger_job(
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
        )

        self.logger.debug("Job server is %s", "active" if _job_server_running() else "inactive")
        if _job_server_maize():
            self.logger.debug("Job server is spawned from maize")
        if not _job_server_running():
            self._restart_jobserver()

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
                self.logger.debug("It werkzz, now monitoring...")

        # Submission success, get the ID and monitor progression
        n_fails = 0
        _, jobid = result.stdout.decode().split()
        while not self.signal.is_set():
            time.sleep(self.query_interval.value)
            match _query_schrodinger_job(jobid):
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

                # Still running or waiting for licenses
                case _SchrodingerJobStatus.RUNNING | _SchrodingerJobStatus.WAITING:
                    self.logger.debug("Running or waiting for licenses...")

                # Timed out due to not enough licenses
                case _SchrodingerJobStatus.LICENSE:
                    self.logger.debug("Timed out due to licenses, retrying...")
                    return _restart(n_fail=_n_failures + 1)

                # Job server died or never started properly
                case _SchrodingerJobStatus.SERVER | _SchrodingerJobStatus.STOPPED:
                    self.logger.warning("Job server communication failure, restarting server")
                    res = self._restart_jobserver()

                    # Job server restart failed, just resubmit whole job (and try to kill zombies)
                    if res.returncode != 0 or _query_schrodinger_job(jobid) in (
                        _SchrodingerJobStatus.SERVER,
                        _SchrodingerJobStatus.UNKNOWN,
                    ):
                        self.logger.warning("Restarting job server unsuccessful, resubmitting job")
                        _kill_associated(jobid, name)
                        return _restart(n_fail=_n_failures + 1)

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
