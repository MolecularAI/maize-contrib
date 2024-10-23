"""Docking with GNINA"""

from functools import partial, reduce
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, FileParameter, Suffix, Input, Output
from maize.utilities.chem import Isomer, IsomerCollection
from maize.utilities.chem.chem import find_mol, load_sdf_library, merge_libraries, save_sdf_library
from maize.utilities.testing import TestRig
from maize.utilities.validation import FileValidator
from maize.utilities.io import Config
from maize.utilities.resources import cpu_count
from maize.utilities.execution import GPU


ScoreType = Literal["default", "ad4_scoring", "dkoes_fast", "dkoes_scoring", "vina", "vinardo"]
CNNScoreType = Literal["none", "rescore", "refinement", "metrorescore", "metrorefine", "all"]
PDBFileType = Annotated[Path, Suffix("pdb", "pdbqt")]


class _GNINA(Node):
    """GNINA base"""

    SCORE_TAGS = ("minimizedAffinity", "CNNscore", "CNNaffinity", "CNN_VS")
    SCORE_TAGS_AGG: tuple[Literal["min", "max"], ...] = ("min", "max", "max", "max")
    PRIMARY_SCORE_TAG = "minimizedAffinity"

    required_callables = ["gnina"]

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to dock"""

    out: Output[list[IsomerCollection]] = Output()
    """Docked molecules with conformations and scores attached"""

    search_range: Parameter[tuple[float, float, float]] = Parameter(default=(15.0, 15.0, 15.0))
    """Range of the search space for docking"""

    autobox_add: Parameter[float] = Parameter(default=4.0)
    """Amount of buffer space to add around the ligand"""

    scoring: Parameter[ScoreType] = Parameter(default="default")
    """Scoring function to use"""

    cnn_scoring: Parameter[CNNScoreType] = Parameter(default="rescore")
    """CNN scoring method to use"""

    exhaustiveness: Parameter[int] = Parameter(default=8)
    """Exhaustiveness of the global search (roughly proportional to time)"""

    n_poses: Parameter[int] = Parameter(default=8)
    """Maximum number of poses to generate"""

    score_only: Flag = Flag(default=False)
    """
    If ``True``, will only score the provided pose without conformational search.
    With this option, neither a reference nor search_center need to be provided.

    """

    n_jobs: Parameter[int] = Parameter(default=cpu_count())
    """The number of CPUs to use per docking run"""

    gpu: Flag = Flag(default=True)
    """Whether to use the GPU for the CNN scoring step"""


class GNINAEnsemble(_GNINA):
    """
    Ensemble docks molecules with GNINA.

    You must specify multiple conformers of the receptor to dock to. Every molecule
    will be docked against all conformations, unless ``score_only`` is set, in which
    case conformer 1 will be scored with molecule 1, 2 with 2, etc.

    See the `repo <https://github.com/gnina/gnina>`_ and [#mcnutt2021]_ for more information.

    References
    ----------
    .. [#mcnutt2021] McNutt, A., Francoeur, P., Aggarwal, R., Masuda, T., Meli, R.,
       Ragoza, M., Sunseri, J. & Koes, D. R. GNINA 1.0: Molecular docking with deep learning.
       J. Cheminformatics 13, 43, (2021).

    """
    tags = {"chemistry", "docking", "scorer", "tagger", "ensemble"}

    inp_ref: Input[list[Isomer]] = Input(optional=True)
    """Reference pose input"""

    ensemble: FileParameter[list[PDBFileType]] = FileParameter(optional=True)
    """Paths to all receptor conformations"""

    ensemble_weights: Parameter[list[float]] = Parameter(optional=True)
    """Optional weights for each ensemble conformation"""

    search_center: Parameter[NDArray[np.float32]] = Parameter(optional=True)
    """Center of the search space for docking"""

    n_parallel: Parameter[int] = Parameter(default=1)
    """
    Number of parallel jobs to run for ensemble docking. Make sure that
    ``n_parallel * n_jobs`` does not exceed the number of available cores,
    or use batch processing.

    """

    def run(self) -> None:
        mols = self.inp.receive()
        protein_confs = self.ensemble.filepath
        weights = np.ones_like(protein_confs, dtype=np.float32) / len(protein_confs)
        if self.ensemble_weights.is_set:
            weights = np.array(self.ensemble_weights.value, dtype=np.float32)
        
        if (refs := self.inp_ref.receive_optional()) is not None:
            use_reference = True
            for i, ref in enumerate(refs):
                ref.to_sdf(Path(f"ref-{i}.sdf"))
        else:
            use_reference = False
            search_centers = self.search_center.value

        inputs = Path("mols.sdf")

        # Get GPU status
        mps_only = False
        gpus = GPU.from_system()
        gpu_ok = any(gpu.free for gpu in gpus)
        if not gpu_ok:
            mps_only = any(gpu.free_with_mps for gpu in gpus)
        self.logger.info("GPU %savailable %s", "not " if not gpu_ok else "", ", MPS required" if mps_only else "")

        if not self.score_only.value:
            save_sdf_library(inputs, mols, split_strategy="none")

        commands = []
        outputs = []
        for i, conf in enumerate(protein_confs):
            if self.score_only.value:
                inputs = Path(f"mols-{i}.sdf")
                mols[i].molecules[0].to_sdf(inputs)

            output = Path(f"output-{i}.sdf")
            command = (
                f"{self.runnable['gnina']} -l {inputs.as_posix()} -r {conf.as_posix()} "
                f"--scoring {self.scoring.value} --cnn_scoring {self.cnn_scoring.value} "
                f"--exhaustiveness {self.exhaustiveness.value} --num_modes {self.n_poses.value} "
                f"--cpu {self.n_jobs.value} --out {output.as_posix()} "
            )

            if use_reference:
                command += f"--autobox_ligand ref-{i}.sdf "
                command += f"--autobox_add {self.autobox_add.value} "

            elif self.score_only.value:
                command += "--score_only "

            else:
                x, y, z = search_centers[i]
                dx, dy, dz = self.search_range.value
                command += f"--center_x {x} --center_y {y} --center_z {z} "
                command += f"--size_x {dx} --size_y {dy} --size_z {dz} "

            # We have the following scenarios for our GPUs:
            # 1) No GPU in the system / user doesn't want GPU -> Run on CPU
            # 2) GPU available but blocked and no MPS -> Run on CPU
            # 3) GPU available but blocked, MPS running -> Run on GPU but use MPS
            # 4) GPU available -> Run on GPU
            if not self.gpu.value or not (gpu_ok or mps_only):
                command += "--no_gpu "
            
            commands.append(command)
            outputs.append(output)

        self.run_multi(
            commands,
            cuda_mps=mps_only and not self.batch_options.is_set,
            n_jobs=self.n_parallel.value,
        )

        libs = [load_sdf_library(output, split_strategy="inchi", renumber=False) for output in outputs]
        for i, lib in enumerate(libs):
            for mol in lib:
                for iso in mol.molecules:
                    for score_tag, agg in zip(self.SCORE_TAGS, self.SCORE_TAGS_AGG):
                        value = float(cast(float, iso.get_tag(score_tag)))
                        iso.add_score(f"{score_tag}-{i}", value, agg=agg)

        mols = reduce(partial(merge_libraries, overwrite_conformers=False), libs)
        for mol in mols:
            for iso in mol.molecules:
                for score_tag, agg in zip(self.SCORE_TAGS, self.SCORE_TAGS_AGG):
                    all_scores = np.array([iso.scores[f"{score_tag}-{i}"] for i, _ in enumerate(outputs)])
                    iso.add_score(score_tag, float(weights @ all_scores), agg=agg)
                iso.primary_score_tag = self.PRIMARY_SCORE_TAG
                iso.set_tag("score_type", "oracle")
                iso.set_tag("origin", self.name)
                self.logger.info(
                    "Parsed isomer '%s', score %s", iso.name or iso.inchi, iso.primary_score
                )
            mol.primary_score_tag = self.PRIMARY_SCORE_TAG

        self.out.send(mols)



class GNINA(_GNINA):
    """
    Docks molecules with GNINA.

    See the `repo <https://github.com/gnina/gnina>`_ and [#mcnutt2021]_ for more information.

    References
    ----------
    .. [#mcnutt2021] McNutt, A., Francoeur, P., Aggarwal, R., Masuda, T., Meli, R.,
       Ragoza, M., Sunseri, J. & Koes, D. R. GNINA 1.0: Molecular docking with deep learning.
       J. Cheminformatics 13, 43, (2021).

    """
    tags = {"chemistry", "docking", "scorer", "tagger"}

    inp_ref: Input[Isomer | str] = Input(optional=True)
    """Reference pose input, or name of a compound"""

    receptor: FileParameter[PDBFileType] = FileParameter(optional=True)
    """Path to the receptor structure"""

    search_center: Parameter[tuple[float, float, float]] = Parameter(optional=True)
    """Center of the search space for docking"""

    blind: Flag = Flag(default=False)
    """
    If ``True``, will attempt blind docking to the full protein,
    you should increase ``exhaustiveness`` in this case.

    """

    def run(self) -> None:
        mols = self.inp.receive()
        protein = self.receptor.filepath
        inputs = Path("mols.sdf")
        output = Path("output.sdf")

        # Get GPU status
        mps_only = False
        gpus = GPU.from_system()
        gpu_ok = any(gpu.free for gpu in gpus)
        if not gpu_ok:
            mps_only = any(gpu.free_with_mps for gpu in gpus)
        self.logger.info("GPU %savailable %s", "not " if not gpu_ok else "", ", MPS required" if mps_only else "")

        command = (
            f"{self.runnable['gnina']} -l {inputs.as_posix()} -r {protein.as_posix()} "
            f"--scoring {self.scoring.value} --cnn_scoring {self.cnn_scoring.value} "
            f"--exhaustiveness {self.exhaustiveness.value} --num_modes {self.n_poses.value} "
            f"--cpu {self.n_jobs.value} --out {output.as_posix()} "
        )

        ref: Isomer | str | None
        if (ref := self.inp_ref.receive_optional()) is not None:
            ref_file = Path("ref.sdf")
            if isinstance(ref, str):
                ref = find_mol(mols, value=ref)
            ref.to_sdf(ref_file)
            command += f"--autobox_ligand {ref_file.as_posix()} "
            command += f"--autobox_add {self.autobox_add.value} "

        elif self.blind.value:
            command += f"--autobox_ligand {protein.as_posix()} "

        elif self.score_only.value:
            command += "--score_only "

        else:
            x, y, z = self.search_center.value
            dx, dy, dz = self.search_range.value
            command += f"--center_x {x} --center_y {y} --center_z {z} "
            command += f"--size_x {dx} --size_y {dy} --size_z {dz} "

        # We have the following scenarios for our GPUs:
        # 1) No GPU in the system / user doesn't want GPU -> Run on CPU
        # 2) GPU available but blocked and no MPS -> Run on CPU
        # 3) GPU available but blocked, MPS running -> Run on GPU but use MPS
        # 4) GPU available -> Run on GPU
        if not self.gpu.value or not (gpu_ok or mps_only):
            command += "--no_gpu "

        save_sdf_library(inputs, mols, split_strategy="none")
        self.run_command(
            command,
            validators=[FileValidator(output)],
            cuda_mps=(mps_only and not self.batch_options.is_set) and self.gpu.value,
            prefer_batch=True,
        )

        mols = load_sdf_library(output, split_strategy="inchi", renumber=False)
        for mol in mols:
            for iso in mol.molecules:
                for score_tag, agg in zip(self.SCORE_TAGS, self.SCORE_TAGS_AGG):
                    iso.add_score_tag(score_tag, agg=agg)
                    for conf in iso.conformers:
                        conf.add_score_tag(score_tag, agg=agg)

                iso.primary_score_tag = self.PRIMARY_SCORE_TAG
                iso.set_tag("score_type", "oracle")
                iso.set_tag("origin", self.name)
                self.logger.info(
                    "Parsed isomer '%s', score %s", iso.name or iso.inchi, iso.primary_score
                )
                iso.addh(add_coords=True)
            mol.primary_score_tag = self.PRIMARY_SCORE_TAG

        self.out.send(mols)


# 1UYD previously published with Icolos (IcolosData/molecules/1UYD)
@pytest.fixture
def protein_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_apo.pdb"


@pytest.fixture
def ligand_path(shared_datadir: Path) -> Path:
    return shared_datadir / "1UYD_ligand.sdf"


class TestSuiteGNINA:
    @pytest.mark.needs_node("gnina")
    def test_GNINA(self, temp_working_dir: Path, protein_path: Path, test_config: Config) -> None:
        """Test GNINA in isolation"""
        rig = TestRig(GNINA, config=test_config)
        params = {
            "search_center": (3.3, 11.5, 24.8),
            "receptor": protein_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        n_atoms_in = mol.molecules[0].n_atoms
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -11.0 < docked[0].primary_score < -7.0
        n_atoms_out = docked[0].molecules[0].n_atoms
        assert n_atoms_in == n_atoms_out

    @pytest.mark.needs_node("gninaensemble")
    def test_GNINA_ensemble(
        self, temp_working_dir: Path, protein_path: Path, test_config: Config
    ) -> None:
        """Test GNINA in isolation"""
        rig = TestRig(GNINAEnsemble, config=test_config)
        params = {
            "search_center": np.array([[3.3, 11.5, 24.8], [3.3, 11.5, 24.8]]),
            "ensemble": [protein_path, protein_path],
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 8
        assert -11.0 < docked[0].primary_score < -7.0
        assert "minimizedAffinity" in docked[0].molecules[0].scores
        assert "minimizedAffinity-0" in docked[0].molecules[0].scores
        assert "minimizedAffinity-1" in docked[0].molecules[0].scores
        assert "CNNaffinity" in docked[0].molecules[0].scores
        assert "CNNaffinity-0" in docked[0].molecules[0].scores
        assert "CNNaffinity-1" in docked[0].molecules[0].scores

    @pytest.mark.needs_node("gnina")
    def test_GNINA_ref(
        self, temp_working_dir: Path, protein_path: Path, ligand_path: Path, test_config: Config
    ) -> None:
        """Test GNINA with reference"""
        rig = TestRig(GNINA, config=test_config)
        params = {
            "receptor": protein_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        ref = Isomer.from_sdf(ligand_path)
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]], "inp_ref": [ref]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -5.0 < docked[0].primary_score < -1.0

    @pytest.mark.needs_node("gnina")
    def test_GNINA_ref_string(
        self, temp_working_dir: Path, protein_path: Path, ligand_path: Path, test_config: Config
    ) -> None:
        """Test GNINA with reference"""
        rig = TestRig(GNINA, config=test_config)
        params = {
            "receptor": protein_path,
            "n_poses": 4,
        }
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        ref = Isomer.from_sdf(ligand_path)
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol, IsomerCollection([ref])]], "inp_ref": [ref.name]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 2
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -5.0 < docked[0].primary_score < -1.0

    @pytest.mark.needs_node("gnina")
    def test_GNINA_score_only(
        self, temp_working_dir: Path, protein_path: Path, ligand_path: Path, test_config: Config
    ) -> None:
        """Test GNINA with reference"""
        rig = TestRig(GNINA, config=test_config)
        params = {"receptor": protein_path, "score_only": True}
        ref = IsomerCollection.from_sdf(ligand_path)
        res = rig.setup_run(parameters=params, inputs={"inp": [[ref]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 1
        assert -5.0 < docked[0].primary_score < 0.0

    @pytest.mark.needs_node("gnina")
    def test_GNINA_blind(
        self, temp_working_dir: Path, protein_path: Path, test_config: Config
    ) -> None:
        """Test GNINA in blind mode (no reference / pocket info)"""
        rig = TestRig(GNINA, config=test_config)
        params = {"receptor": protein_path, "n_poses": 4, "blind": True}
        mol = IsomerCollection.from_smiles("Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC")
        mol.embed()
        res = rig.setup_run(parameters=params, inputs={"inp": [[mol]]})
        docked = res["out"].get()
        assert docked is not None
        assert len(docked) == 1
        assert docked[0].scored
        assert docked[0].molecules[0].n_conformers == 4
        assert -11.0 < docked[0].primary_score < -7.0
