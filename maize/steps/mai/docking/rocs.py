"""ROCS shape-matching implementation, original code by Michael Dodds"""

# pylint: disable=import-outside-toplevel, import-error

from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from numpy.typing import NDArray
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, FileParameter, Suffix, Input, Output
from maize.utilities.chem.chem import ChemistryException, Isomer
from maize.utilities.testing import TestRig
from maize.utilities.io import Config

from maize.utilities.chem import IsomerCollection


_SimMeasureType = Literal["Tanimoto", "RefTversky", "FitTversky"]
_SCORE_METHODS = {
    "Tanimoto": ("GetTanimoto", "GetColorTanimoto"),
    "RefTversky": ("GetRefTversky", "GetRefColorTversky"),
    "FitTversky": ("GetFitTversky", "GetFitColorTversky"),
}
_SCORE_COMBO = {
    "Tanimoto": "OEHighestTanimotoCombo",
    "RefTversky": "OEHighestRefTverskyCombo",
    "FitTversky": "OEHighestFitTverskyCombo",
}


# Potentially useful if we need to do more stuff with OpenEye in the future:
# https://gist.github.com/bannanc/810ccc4636b930a4522636baab1965a6
def _isomer2oe(isomer: Isomer) -> Any:
    """Convert a maize isomer to an ``OEMol`` object"""
    from openeye import oechem

    sdf = isomer.to_mol_block()
    ims = oechem.oemolistream()
    ims.SetFormat(oechem.OEFormat_SDF)
    ims.openstring(sdf)
    return oechem.OEMol(next(ims.GetOEMols()))


def _oe2isomer_conf(oemol: Any, isomer: Isomer) -> None:
    """Convert an ``OEMol`` object to a conformer added to an `Isomer`"""
    from openeye import oechem

    # OpenEye adds pseudo-atoms during ROCS (donor, acceptor, etc.), we remove
    # these and add explicit hydrogens before conversion so RDKit doesn't choke
    for atom in oemol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            oemol.DeleteAtom(atom)
    oechem.OEPlaceHydrogens(oemol)

    # This procedure saves us from having to write to a file
    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_SDF)
    oms.openstring()
    oechem.OEWriteMolecule(oms, oemol)
    isomer.update_conformers_from_mol_block(oms.GetString().decode("UTF-8"))


def _gen_conf(oemol: Any, omega: Any, max_stereo: int = 0) -> None:
    """Generate enantiomers using OEFlipper"""
    from openeye import oeomega, oechem

    # OEMol object, max. number of stereocenters, force flip, enum nitrogen
    enantiomers = [
        oechem.OEMol(isomer)
        for isomer in oeomega.OEFlipper(oemol.GetActive(), max_stereo, False, True)
    ]
    first = enantiomers[0]
    omega.Build(first)
    oemol = oechem.OEMol(first.SCMol())
    oemol.DeleteConfs()

    # Add conformers to OEMol object
    for enantiomer in enantiomers:
        omega.Build(enantiomer)
        for conf in enantiomer.GetConfs():
            oemol.NewConf(conf)


def _prepare_overlay(shape_query: Path) -> Any:
    """Prepares the shape query and returns an overlay"""
    from openeye import oeshape

    query = oeshape.OEShapeQuery()
    oeshape.OEReadShapeQuery(shape_query.as_posix(), query)
    overlay = oeshape.OEOverlay()
    overlay.SetupRef(query)
    return overlay


def _score(
    oemol: Any,
    similarity_measure: _SimMeasureType,
    overlay: Any,
    shape_weight: float = 0.5,
    color_weight: float = 0.5,
) -> tuple[Any, float]:
    """Calculate the ROCS score"""
    from openeye import oeshape

    score = oeshape.OEBestOverlayScore()
    combo = getattr(oeshape, _SCORE_COMBO[similarity_measure])
    overlay.BestOverlay(score, oemol, combo())
    shape, color = (getattr(score, met)() for met in _SCORE_METHODS[similarity_measure])
    return score, shape_weight * shape + color_weight * color


class ROCS(Node):
    """
    Performs ROCS shape-match scoring [#grant1996]_.

    Notes
    -----
    Requires a maize environment with ``openeye-toolkit`` installed. OpenEye in turn
    requires the OE_LICENSE environment variable to be set to a valid license file.

    References
    ----------
    .. [#grant1996] Grant, J. A., Gallardo, M. A. & Pickup, B. T. A fast method of
       molecular shape comparison: A simple application of a Gaussian description of
       molecular shape. Journal of Computational Chemistry 17, 1653-1666 (1996).

    See also the `full list of related publications
    <https://docs.eyesopen.com/applications/rocs/pub.html>`_.

    """
    required_packages = ["openeye"]

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to be scored"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with conformers best matching the query"""

    out_scores: Output[NDArray[np.float32]] = Output()
    """Score output"""

    query: FileParameter[Annotated[Path, Suffix("sq")]] = FileParameter()
    """Reference query molecule"""

    max_stereo: Parameter[int] = Parameter(default=10)
    """Maximum number of stereocenters to be enumerated in molecule"""

    max_confs: Parameter[int] = Parameter(default=200)
    """Maximum number of conformers generated per stereoisomer"""

    energy_window: Parameter[int] = Parameter(default=10)
    """Difference between lowest and highest energy conformer"""

    similarity_measure: Parameter[_SimMeasureType] = Parameter(default="Tanimoto")
    """Similarity between reference and molecule"""

    color_weight: Parameter[float] = Parameter(default=0.5)
    """Weight applied to the color-matching score"""

    shape_weight: Parameter[float] = Parameter(default=0.5)
    """Weight applied to the shape-matching score"""

    scores_only: Flag = Flag(default=True)
    """Whether to only output scores, without poses"""

    strict: Flag = Flag(default=False)
    """If ``True`` will fail and raise an exception when failing to score a molecule"""

    gpu: Flag = Flag(default=True)
    """Whether to use the GPU"""

    def run(self) -> None:
        from openeye import oechem, oeomega, oequacpac, oeshape

        # Conformer generation options
        omega_options = oeomega.OEOmegaOptions()
        omega_options.SetStrictStereo(False)
        omega_options.SetEnergyWindow(self.energy_window.value)
        omega_options.SetMaxConfs(self.max_confs.value)
        omega_options.GetTorDriveOptions().SetUseGPU(self.gpu.value)
        omega = oeomega.OEOmega(omega_options)

        overlay = _prepare_overlay(self.query.filepath)
        prep = oeshape.OEOverlapPrep()

        # Normalize weights in case they don't add up to 1.0
        shape_weight = self.shape_weight.value
        color_weight = self.color_weight.value
        weight_sum = shape_weight + color_weight
        shape_weight, color_weight = shape_weight / weight_sum, color_weight / weight_sum

        scores = []
        mols = self.inp.receive()
        for mol in mols:
            if len(mol.molecules) > 1:
                self.logger.warning(
                    "Molecule '%s' has more than one isomer. ROCS performs it's own "
                    "conformer generation and will ignore all but the first isomer.",
                    mol.smiles,
                )

            isomer = mol.molecules[0]
            oemol = _isomer2oe(isomer)
            oequacpac.OEGetReasonableProtomer(oemol)

            # Create conformers
            _gen_conf(oemol, omega=omega, max_stereo=self.max_stereo.value)

            # Non-zero returncode indicates an error
            if omega.Build(oemol):
                if self.strict.value:
                    raise ChemistryException(f"Omega failed to build '{isomer.inchi}'")
                self.logger.warning("Omega failed to build '%s'", isomer.inchi)
                score = np.nan
                scores.append(score)
                isomer.scores = np.array([score])
                continue

            prep.Prep(oemol)
            scorer, result = _score(
                oemol,
                similarity_measure=self.similarity_measure.value,
                overlay=overlay,
                shape_weight=shape_weight,
                color_weight=color_weight,
            )
            self.logger.info("Shape matched '%s' with a score of %s", isomer.inchi, result)
            scores.append(result)
            isomer.scores = np.array([result])
            if not self.scores_only.value:
                docked = oechem.OEGraphMol(
                    oemol.GetConf(oechem.OEHasConfIdx(scorer.GetFitConfIdx()))
                )
                _oe2isomer_conf(docked, isomer)

        self.out_scores.send(np.array(scores))
        if not self.scores_only.value:
            self.out.send(mols)


# Published COX-2 inhibitor: https://go.drugbank.com/drugs/DB03477
@pytest.fixture
def shape_query(shared_datadir: Path) -> Path:
    return shared_datadir / "S58.sq"


@pytest.fixture
def example_mol(shared_datadir: Path) -> IsomerCollection:
    return IsomerCollection.from_sdf(shared_datadir / "S58.sdf")


def test_rocs(
    shape_query: Path,
    example_mol: IsomerCollection,
    test_config: Config,
    temp_working_dir: Path,
) -> None:
    """Test our step in isolation"""
    other_mol = IsomerCollection.from_smiles("CCCO")
    other_mol.embed()
    mols = [example_mol, other_mol]
    rig = TestRig(ROCS, config=test_config)
    res = rig.setup_run(
        inputs={"inp": [mols]},
        parameters={"query": shape_query, "scores_only": False},
    )
    docked = res["out"].get()
    scores = res["out_scores"].get()
    assert docked is not None
    assert scores is not None
    assert len(docked) == 2
    assert len(scores) == 2
    for mol in docked:
        assert mol.scored
        for iso in mol.molecules:
            assert iso.scores is not None
            assert 0 < iso.scores[0] < 1