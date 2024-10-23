"""Tests for docking workflows"""

from pathlib import Path
from typing import Any

import pytest

from maize.graphs.mai.dock import dock_single


@pytest.fixture
def smiles_1uyd() -> str:
    return "Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC"


@pytest.fixture
def receptor_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_fixed.pdbqt"


@pytest.fixture
def search_center() -> tuple[float, float, float]:
    return (3.3, 11.5, 24.8)


@pytest.fixture
def test_config() -> Path:
    return Path("devtools/test-config.toml")


def test_dock_single(
    mocker: Any,
    smiles_1uyd: str,
    receptor_path: Path,
    search_center: tuple[float, float, float],
    test_config: Path,
) -> None:
    result_file = Path("out.sdf")
    mocker.patch(
        "sys.argv",
        [
            "testing",
            "--config",
            test_config.as_posix(),
            "--smiles",
            smiles_1uyd,
            "--receptor",
            receptor_path.as_posix(),
            "--search_center",
            *(str(c) for c in search_center),
            "--output",
            result_file.as_posix(),
        ],
    )
    dock_single()
    assert result_file.exists()
