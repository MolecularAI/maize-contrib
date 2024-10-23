"""Interface to Lomap"""

# pylint: disable=import-outside-toplevel, import-error
from pathlib import Path

import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output
from maize.utilities.validation import FileValidator
from maize.utilities.testing import TestRig
from maize.utilities.chem import Isomer
from maize.utilities.io import Config


class Lomap(Node):
    """
    Interface to the Lead Optimization MAPper (Lomap).

    See [#liu2013]_.

    Notes
    -----
    Lomap can be installed into a copy of a maize environment from
    `here <https://github.com/OpenFreeEnergy/Lomap>`_.

    References
    ----------
    .. [#liu2013] Liu, S. et al. Lead Optimization Mapper: Automating free energy
       calculations for lead optimization. J Comput Aided Mol Des 27, (2013)
       `DOI <https://doi.org/10.1007/s10822-013-9678-y>`_

    """
    tags = {"chemistry", "fep"}

    required_callables = ["lomap"]
    """
    scripts
        Requires the ``lomap`` script.

    """

    inp: Input[list[Isomer]] = Input()
    """List of molecules to map"""

    inp_reference: Input[Isomer] = Input(optional=True)
    """Reference molecule for a star-map"""

    out: Output[dict[tuple[str, str], dict[int, int]]] = Output()
    """Edges with potential atom-mappings"""

    def run(self) -> None:
        mols = self.inp.receive()
        ref = None
        # Lomap wants all molecules in separate SDF files
        mapdir = Path("map")
        mapdir.mkdir()
        for isomer in mols:
            isomer.to_sdf(mapdir / f"{isomer.inchi}.sdf")

        command = f"{self.runnable['lomap']} -c 0.0 "
        
        # Supplying a reference implies a star-map (for now)
        if self.inp_reference.ready():
            ref = self.inp_reference.receive()
            ref.to_sdf(mapdir / f"{ref.inchi}.sdf")
            command += f"-r -b {ref.inchi}.sdf "
        command += mapdir.as_posix()
        output = Path("out_score_with_connection.txt")
        self.run_command(command, verbose=True, validators=[FileValidator(output)])

        # Pandas has problems with this file due to the
        # incomplete header, parsing manually is easier
        with output.open() as read:
            raw = read.readlines()[1:]

        data: dict[tuple[str, str], dict[int, int]] = {}
        for line in raw:
            # Filenames are the INCHI keys, the mapping contains the atom re-mapping
            (_, _, file_i, file_j, _, _, _, _, *mapping) = (tok.strip() for tok in line.split(","))

            # Convert raw mapping to real dictionary with indices
            atom_map: dict[int, int] = {}
            for ma in mapping:
                if ":" not in ma:
                    break
                i, j, *_ = (int(atom) for atom in ma.split(":"))
                atom_map[int(i)] = int(j)
            data[(file_i.strip(".sdf"), file_j.strip(".sdf"))] = atom_map

        self.out.send(data)


# 1UYD ligands (IcolosData)
@pytest.fixture
def smiles() -> list[str]:
    return [
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cc(OC)c(OC)c(c3)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC",
        "Nc1nc(F)nc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc(cc3)cc(c34)OCO4",
    ]


@pytest.mark.needs_node("lomap")
def test_lomap(temp_working_dir: Path, smiles: list[str], test_config: Config) -> None:
    """Test our step in isolation"""
    mols = [Isomer.from_smiles(smi) for smi in smiles]
    rig = TestRig(Lomap, config=test_config)
    res = rig.setup_run(inputs={"inp": [mols[1:]], "inp_reference": [mols[0]]})
    data = res["out"].get()
    refnum = 0
    assert data is not None
    for (a, b) in data:
        refnum += mols[0].inchi in (a, b)
    assert refnum == 5

