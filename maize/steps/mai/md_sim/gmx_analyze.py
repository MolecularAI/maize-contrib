"""Analyzing trajectories from GROMACS"""

# pylint: disable=import-outside-toplevel, import-error
from inspect import Parameter
from pathlib import Path
from typing import Annotated

import pytest
import numpy as np
from numpy.typing import NDArray

from maize.core.node import Node
from maize.core.interface import Input, Output, FileParameter, Parameter, Suffix
from maize.utilities.testing import TestRig
from maize.utilities.chem import IsomerCollection


class Analyze_GMX_MD(Node):
    """
    Analyzes GROAMCS trajectories using gmx analysis tools and combining results
    into a feature table (columns = feature, rows = system)
    """

    # required_callables = ["gromacs"]

    inp: FileParameter[Path] = FileParameter()
    """Path to folder that contains all system subfolders with MD trajectories"""

    replicas: Parameter[int | None] = Parameter()
    """Number of replicas systems were run"""

    # out: Parameter[NDArray[np.float32]] = Parameter()
    # """Table xxx """

    def run(self) -> None:
        root_path = self.inp.receive()

        # from chatgpt, check later
        system_paths = [
            x
            for x in cwd.iterdir()
            if x.is_dir() and any((f.suffix == ".trr" or f.suffix == ".xtc") for f in x.glob("*"))
        ]
