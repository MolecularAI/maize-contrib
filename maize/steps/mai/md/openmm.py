"""Nodes to run simple OpenMM simulations"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, FileParameter, Flag, Suffix, Input, Output
from maize.utilities.chem import Isomer, IsomerCollection
from maize.utilities.testing import TestRig
from maize.utilities.io import Config
from maize.utilities.resources import cpu_count


OPENMM_SCRIPT = Path(__file__).parent / "scripts" / "mm-simple.py"

# Maximum number of processes when running with MPS (?)
MAX_JOBS_PER_GPU = 16


# TODO Maybe do this with the `saveState` and `loadState` methods of the simulation instance,
# that way we could have a super minimal runscript that is agnostic to the type of simulation
# (at least to some degree)
class PoseStability(Node):
    """
    Run a molecular dynamics simulation of a small molecule
    in a protein pocket to assess pose stability.

    """
    tags = {"chemistry", "scorer", "tagger", "md"}

    required_packages = ["mdtraj"]
    """
    mdtraj
        Requires ``mdtraj`` for trajectory analysis

    """

    required_callables = ["openmm_script"]
    """
    openmm_script
        OpenMM python script for protein-ligand MD

    """

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    receptor: FileParameter[Annotated[Path, Suffix("pdb")]] = FileParameter()
    """Protein structure"""

    temperature: Parameter[float] = Parameter(default=298.15)
    """Temperature in Kelvin to use for all simulations"""

    ion_concentration: Parameter[float] = Parameter(default=0.15)
    """Ion concentration in molar"""

    neutralize: Flag = Flag(default=True)
    """Whether to neutralize the system"""

    equilibration_length: Parameter[int] = Parameter(default=200)
    """Length of equilibration simulation in ps"""

    production_length: Parameter[int] = Parameter(default=1000)
    """Length of production simulation in ps"""

    solvent: Parameter[Literal["tip3p", "spce", "tip4pew", "tip5p"]] = Parameter(default="tip3p")
    """Water model to use"""

    padding: Parameter[float] = Parameter(default=1.2)
    """Minimum distance of the solute to the box edge"""

    box: Parameter[Literal["cubic", "dodecahedron", "octahedron"]] = Parameter(
        default="dodecahedron"
    )
    """Box type to use"""

    platform: Parameter[Literal["CUDA", "CPU", "OpenCL", "Reference"]] = Parameter(default="CUDA")
    """The OpenMM compute platform"""

    n_threads: Parameter[int] = Parameter(default=1)
    """Number of threads per launched simulation"""

    def run(self) -> None:
        import mdtraj as md

        mols = self.inp.receive()
        protein = self.receptor.filepath

        commands = []
        for mol in mols:
            for iso in mol.molecules:
                mol_input = Path(f"iso-{iso.inchi}.sdf")
                mol_output = Path(f"out-{iso.inchi}.pdb")
                iso.addh()
                iso.to_sdf(mol_input)
                command = (
                    f"{self.runnable['openmm_script']} --protein {protein} "
                    f"--mol {mol_input} "
                    f"--output {mol_output} "
                    f"--temperature {self.temperature.value} "
                    f"--ion-conc {self.ion_concentration.value} "
                    f"--eq-length {self.equilibration_length.value} "
                    f"--prod-length {self.production_length.value} "
                    f"--solvent {self.solvent.value} "
                    f"--box {self.box.value} "
                    f"--padding {self.padding.value} "
                    f"--n-threads {self.n_threads.value} "
                )
                if self.neutralize.value:
                    command += "--neutralize "
                if self.platform.is_set:
                    command += f"--platform {self.platform.value}"
                commands.append(command)

        mps = (
            self.platform.value == "CUDA"
            and len([iso for mol in mols for iso in mol.molecules]) > 1
        )

        # We can do many parallel simulations on the same GPU with CUDA MPS
        n_jobs = (
            min(MAX_JOBS_PER_GPU, cpu_count() // self.n_threads.value)
            if self.platform.value in ("CUDA", "OpenCL")
            else cpu_count() // self.n_threads.value
        )
        self.run_multi(commands, n_jobs=n_jobs, cuda_mps=mps)

        for mol in mols:
            for iso in mol.molecules:
                output = Path(f"out-{iso.inchi}.pdb")
                if not output.exists():
                    iso.add_score("final_rmsd", np.nan, agg="min")
                    self.logger.warning("Pose stability estimate failed for %s", iso.name)
                    continue

                traj = md.load_pdb(output)
                inds = traj.topology.select("resname UNK")
                rmsd = md.rmsd(traj, traj[0], atom_indices=inds)
                iso.set_tag("final-rmsd", rmsd[-1])
                iso.add_score("final_rmsd", rmsd[-1], agg="min")
                self.logger.info(
                    "RMSD for %s after %s ps: %s nm",
                    iso.name,
                    self.production_length.value,
                    rmsd[-1],
                )

        self.out.send(mols)


@pytest.fixture
def protein(shared_datadir: Path) -> Annotated[Path, Suffix("pdb")]:
    return shared_datadir / "tnks.pdb"


@pytest.fixture
def mol(shared_datadir: Path) -> Isomer:
    return Isomer.from_sdf(shared_datadir / "ref.sdf")


class TestSuiteOpenMM:
    @pytest.mark.needs_node("posestability")
    def test_PoseStability(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein: Annotated[Path, Suffix("pdb")],
        mol: Isomer,
    ) -> None:
        rig = TestRig(PoseStability, config=test_config)
        res = rig.setup_run(
            inputs={
                "inp": [[IsomerCollection([mol])]],
            },
            parameters={
                "receptor": protein,
                "equilibration_length": 20,
                "production_length": 100,
            },
        )
        out = res["out"].get()
        assert out is not None
        assert len(out) == 1
        assert "final_rmsd" in out[0].molecules[0].scores
        assert out[0].molecules[0].scores["final_rmsd"] > 0.0
