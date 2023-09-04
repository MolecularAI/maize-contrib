"""Schrodinger Ligprep prepares 3D small molecule conformers and isomers"""

# pylint: disable=import-outside-toplevel, import-error

from collections.abc import Sequence
import os
from pathlib import Path
import re
from subprocess import CompletedProcess
import time

from maize.core.node import Node
from maize.core.interface import Parameter
from maize.utilities.execution import CommandRunner, JobResourceConfig
from maize.utilities.validation import Validator


SCHRODINGER_LICENSE = "SCHROD_LICENSE_FILE"


def has_license() -> bool:
    """``True`` if the system has a Schrodinger license, ``False`` otherwise."""
    return not ((loc := os.environ.get(SCHRODINGER_LICENSE, "")) == "" or not Path(loc).exists())


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

    def __init__(self, parent: Node, pre: str | list[str]) -> None:
        self.pre_execution = pre
        self.parent = parent
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

    def query(self) -> None:
        """Queries the license server for available tokens"""
        out = CommandRunner().run("licadmin STAT", pre_execution=self.pre_execution)

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
            ``True`` if licenses are available, ``False`` otherwise

        """
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

        return


class Schrodinger(Node, register=False):
    SCHRODINGER_PRE = "jsc local-server-start"

    guard: TokenGuard

    n_jobs: Parameter[int] = Parameter(default=1)
    """Number of jobs to spawn"""

    host: Parameter[str] = Parameter(default="localhost")
    """Host to use for job submission"""

    def prepare(self) -> None:
        super().prepare()
        self.guard = TokenGuard(parent=self, pre=self.SCHRODINGER_PRE)
        return

    # Always make sure we're running the preliminary commands
    def run_command(
        self,
        command: str | list[str],
        validators: Sequence[Validator] | None = None,
        verbose: bool = False,
        raise_on_failure: bool = True,
        command_input: str | None = None,
        pre_execution: str | list[str] | None = None,
        batch_options: JobResourceConfig | None = None,
        timeout: float | None = None
    ) -> CompletedProcess[bytes]:
        if pre_execution is not None:
            if isinstance(pre_execution, str):
                pre_execution = pre_execution.split()
            pre_execution.extend(["&&", self.SCHRODINGER_PRE])
        else:
            pre_execution = self.SCHRODINGER_PRE
        return super().run_command(
            command,
            validators,
            verbose,
            raise_on_failure,
            command_input,
            pre_execution,
            batch_options,
            timeout,
        )
