"""Pytest fixtures"""

from pathlib import Path
from typing import Any
import pytest

from maize.utilities.io import Config


def pytest_addoption(parser: Any) -> None:
    parser.addini(
        "config",
        help="maize configuration for test suite",
        type="paths",
        default=[Path("test-config.toml")],
    )


@pytest.fixture
def temp_working_dir(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def default_config() -> Config:
    return Config()


@pytest.fixture
def test_config(request: Any) -> Config:
    config = Config()
    file = request.config.getini("config")[0]
    print(Path().absolute().as_posix())
    config.update(file)
    return config


@pytest.fixture(autouse=True)
def skip_by_config(request: Any, test_config: Config) -> None:
    if request.node.get_closest_marker("needs_node"):
        if (
            name := request.node.get_closest_marker("needs_node").args[0].lower()
        ) not in test_config.nodes:
            pytest.skip(f"Skipped due to missing '{name}' config entry")


@pytest.fixture
def example_smiles() -> list[str]:
    return [
        "Nc1ncnc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "Nc1nc(F)nc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3",
        "CC(C)NCCCn(c(c12)nc(F)nc2N)c(n1)Cc(c3)c(I)cc(c34)OCO4",
        "O1COc(c12)cc(Br)c(c2)Cc(nc(n34)c(N)ncc3)c4NCc5ccccc5",
    ]

@pytest.fixture
def evil_example_smiles() -> list[str]:
    return [
        "Nc1ncnc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "Nc1nc(F)nc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "[B]=C",
    ]
