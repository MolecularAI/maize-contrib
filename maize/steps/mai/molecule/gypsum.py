"""GypsumDL prepares 3D small molecule conformers and isomers"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path

import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Flag
from maize.utilities.testing import TestRig
from maize.utilities.validation import SuccessValidator
from maize.utilities.resources import cpu_count
from maize.utilities.execution import ProcessError
from maize.utilities.chem import IsomerCollection, save_smiles
from maize.utilities.io import Config


DEFAULT_FILE_NAME = "untitled_line_{0}__input{0}.sdf"
FAILED_SMILES_FILE = "gypsum_dl_failed.smi"


class Gypsum(Node):
    """
    Converts SMILES codes into a set of 3D molecules using Gypsum-DL.

    See [#ropp2019]_ for more details. 3D embedding can fail,
    and in those cases it falls back on RDKit.

    Notes
    -----
    The implementation in this node does not use the MPI capabilities of Gypsum,
    and simply installing ``MPI4PY`` can cause problems executing this step on some
    HPC systems. So it might be better to simply not install it for this use case.

    References
    ----------
    .. [#ropp2019] Ropp, P.J., Spiegel, J.O., Walker, J.L. et al. Gypsum-DL: an
       open-source program for preparing small-molecule libraries for
       structure-based virtual screening. J Cheminform 11, 34 (2019).
       `DOI <https://doi.org/10.1186/s13321-019-0358-3>`_

    See Also
    --------
    :class:`~maize.steps.mai.molecule.Smiles2Molecules` :
        A simple, fast, and less accurate alternative to
        Gypsum, using RDKit embedding functionality.

    """

    required_callables = ["gypsum"]

    inp: Input[list[str]] = Input()
    """SMILES input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    n_variants: Parameter[int] = Parameter(default=1)
    """Maximum number of variants to generate"""

    thoroughness: Parameter[int] = Parameter(default=3)
    """
    Multiplier for the number of sampled conformers to
    evaluate energies. Higher numbers will increase the
    computational cost by performing more UFF energy evaluations.

    """

    ph_range: Parameter[tuple[float, float]] = Parameter(default=(6.4, 8.4))
    """The pH range in which to generate variants (min, max)"""

    use_filters: Flag = Flag(default=True)
    """Whether to use additional substructure filters from the Durrant lab"""

    n_jobs: Parameter[int] = Parameter(default=cpu_count())
    """Number of parallel processes to use"""

    timeout: Parameter[int] = Parameter(default=5)
    """Timeout per SMILES in seconds, will attempt an RDKit embedding after"""

    def run(self) -> None:
        smiles = [smi.strip() for smi in self.inp.receive()]
        smiles_path = Path("input.smi")
        save_smiles(smiles_path, smiles)
        command = (
            f"{self.runnable['gypsum']} --source {smiles_path.as_posix()} "
            f"--max_variants_per_compound {self.n_variants.value} "
            f"--thoroughness {self.thoroughness.value} --separate_output_files "
            f"--min_ph {self.ph_range.value[0]} --max_ph {self.ph_range.value[1]} "
            f"--job_manager multiprocessing --num_processors {self.n_jobs.value} "
        )
        if self.use_filters.value:
            command += "--use_durrant_lab_filters"

        # With our settings Gypsum produces one SDF file per SMILES,
        # each of which can have one or more isomers / conformers
        res = self.run_command(
            command,
            verbose=True,
            validators=[SuccessValidator("Finished Gypsum-DL")],
            timeout=10 + len(smiles) * self.timeout.value,
            raise_on_failure=False,
        )

        failed = set()
        if res.returncode == 130:  # Timeout
            self.logger.warning("Timed out during embedding")
            failed = set(smiles)
        elif res.returncode > 0:
            raise ProcessError("Gypsum failed for SMILES: %s", smiles)

        # Gypsum can fail to embed certain SMILES, but helpfully writes out those separately
        if Path(FAILED_SMILES_FILE).exists():
            self.logger.info("Found failed SMILES file")
            with Path(FAILED_SMILES_FILE).open() as failed_file:
                failed = {smi.split()[0] for smi in failed_file.readlines()}
                self.logger.info("Failed SMILES:\n'%s'", "\n".join(failed))

        mols = []
        for i, smi in enumerate(smiles):
            gypsum_index = i + 1
            file = Path(DEFAULT_FILE_NAME.format(gypsum_index))
            self.logger.debug("Checking SMILES '%s'", smi)
            if smi in failed:
                self.logger.warning(
                    "Skipping failed embedding for SMILES '%s', falling back to RDKit", smi
                )
                mol = IsomerCollection.from_smiles(smi)
                mol.embed()
                if any(isomer.n_conformers == 0 for isomer in mol.molecules):
                    self.logger.warning("Coordinate generation for isomer '%s' failed", smi)

            # We already check for failed embeddings so this shouldn't really happen
            elif not file.exists() or file.stat().st_size == 0:
                raise FileNotFoundError(
                    f"Gypsum output for '{smi}' at '{file.as_posix()}' not found or empty"
                )

            # All good!
            else:
                mol = IsomerCollection.from_sdf(file)
                mol.smiles = smi
                for isomer in mol.molecules:
                    isomer.name = isomer.inchi

            mols.append(mol)

        self.out.send(mols)


class TestSuiteGypsum:
    @pytest.mark.needs_node("gypsum")
    def test_Gypsum(
        self, temp_working_dir: Path, test_config: Config, example_smiles: list[str]
    ) -> None:
        rig = TestRig(Gypsum, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [example_smiles]},
            parameters={"n_variants": 2},
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == len(example_smiles)
        assert mols[0].molecules[0].n_conformers == 1
        assert mols[0].molecules[0].charge <= 2
        assert 51 <= mols[0].molecules[0].n_atoms <= 53
        for mol in mols:
            assert mol.n_isomers <= 2
            assert not mol.scored
