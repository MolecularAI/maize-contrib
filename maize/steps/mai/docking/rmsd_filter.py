"""RMSD filtering of docking results"""

from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, cast
from typing_extensions import assert_never

import pytest
import numpy as np
from numpy.typing import NDArray

from maize.core.node import Node
from maize.core.interface import Parameter, FileParameter, Suffix, Input, Output, Flag
from maize.utilities.chem.chem import (
    Isomer,
    IsomerCollection,
    rmsd as chemrmsd,
)
from maize.utilities.testing import TestRig


T_arr_float = TypeVar("T_arr_float", NDArray[np.float32], float)


def score_combine(score1: T_arr_float, score2: T_arr_float, weight: float) -> T_arr_float:
    """Combines two normalized scores as geometric mean"""
    return cast(T_arr_float, np.sqrt(weight * np.square(score1 - 1) + np.square(score2 - 1)))


class RMSDFilter(Node):
    """
    Charge filtering for isomers and RMSD filtering for conformers.

    Only isomers with target charge pass filter. For each isomer, only conformers
    that minmize RMSD to a given reference ligand are considered. If several isomers
    with target charge remain after charge filtering, either the isomer with smallest
    RMSD or lowest docking score pass through the filter. At the end, only one isomer
    with one conformer (or none) per SMILES pass the filter.

    """

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules with isomers and conformations (from single SMILES) to filter"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with single isomer and conformer after filtering"""

    ref_lig: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter()
    """Path to the reference ligand"""

    target_charge: Parameter[int] = Parameter(default=0)
    """Only isomers with this total charge pass filter"""

    reference_charge_type: Parameter[Literal["ref", "target", "no"]] = Parameter(default="target")
    """
    If 'ref' is given then the charge of the reference ligand is the target charge.
    If 'target' is given, the charge specified under ``target_charge`` is used. If
    'no' is given, every isomer charge is accepted.

    """

    strict_target_charge: Flag = Flag(default=True)
    """
    If true and no isomer with target charge is found, an empty isomer list passes
    the filter. This is useful for RBFE calculations where FEP edges with changes
    in charge are unsuitable. If false and no isomer with target charge is found,
    accept any other isomer charge. This is useful for a standard REINVENT run
    where for each SMILES a conformation is passing the filter.

    """

    isomer_filter: Parameter[Literal["dock", "rmsd", "combo"]] = Parameter(default="dock")
    """
    If after filtering out isomers with wrong charge more than one isomer remain 
    pass isomer with lowest docking score when set to 'dock', pass isomer with
    lowest rmsd when set to 'rmsd' or pass isomer with lowest combined score when
    set to 'combo'.

    """

    conformer_combo_filter: Flag = Flag(default=True)
    """
    If set to 'True', rmsd and docking score are combined to filter the best conformer
    for each isomer. Otherwise, only RMSD is used to find the best conformer.

    """

    def run(self) -> None:
        mols = self.inp.receive()

        # load in reference ligand
        ref_mol = Isomer.from_sdf(self.ref_lig.filepath)

        # determine the target isomer charge
        if self.reference_charge_type.value == "target":
            charge = self.target_charge.value
            self.logger.info("Target charge explicitly given as %s", self.target_charge.value)
        elif self.reference_charge_type.value == "ref":
            charge = ref_mol.charge
            self.logger.info("Target charge from reference ligand is %s", charge)
        else:
            charge = None  # this means that any isomer charge is acceptable
            self.logger.info("Every isomer charge is accepted")

        if self.strict_target_charge.value and charge is not None:
            self.logger.info("Isomers with charges other than target charge won't pass filter")

        for mol in mols:
            # check if there is an isomer with acceptable charge
            # if not, behaviour depends on setting of strict_target_charge
            good_isocharge = any(isomer.charge == charge for isomer in mol.molecules)

            if not good_isocharge and charge is not None:
                self.logger.info("For molecule %s no isomer with target charge found!", mol)
                if not self.strict_target_charge.value:
                    charge = None

            # find best docking score and rmsd of all isomers and conformers in mol
            # needed for a combo score only
            rmsd_iso_min: float = 0.0
            dock_iso_min: float = 0.0
            if self.isomer_filter.value == "combo":
                rmsd_iso_min = min(chemrmsd(isomer, ref_mol).min() for isomer in mol.molecules)
                dock_iso_min = min(
                    isomer.scores.min() for isomer in mol.molecules if isomer.scores is not None
                )

            # find most suitable isomer
            best_iso_score = np.inf
            best_iso = None
            for isomer in mol.molecules[:]:
                # check if isomer has scores and correct charge
                if isomer.scores is not None and (isomer.charge == charge or charge is None):
                    # get best rmsd and docking score for all conformers
                    rmsd = chemrmsd(isomer, ref_mol)
                    rmsd_conf_min = np.min(rmsd)
                    dock_conf_min = np.min(isomer.scores)

                    # combine scores if conformer_combo_filter is set
                    if self.conformer_combo_filter.value:
                        conf_score = score_combine(
                            isomer.scores / dock_conf_min,
                            rmsd / rmsd_conf_min,
                            100,
                        )
                    else:
                        conf_score = rmsd

                    # select best conformer
                    min_conf_idx = np.argmin(conf_score)
                    min_rmsd: float = rmsd[min_conf_idx]
                    min_dock: float = isomer.scores[min_conf_idx]
                    min_conf = isomer.conformers[min_conf_idx]

                    # To avoid an isomer with empty conformers it is necessary
                    # to first add the best conformer and then to delete all the previous
                    isomer.clear_conformers()
                    isomer.add_conformer(min_conf)
                    # for _ in range(isomer.n_conformers - 1):
                    #     isomer.remove_conformer(0)
                    isomer.scores = np.array([min_dock])

                    # check which isomer has lowest score
                    if self.isomer_filter.value == "dock":
                        score = min_dock
                    elif self.isomer_filter.value == "rmsd":
                        score = min_rmsd
                    elif self.isomer_filter.value == "combo":
                        # normalizing using best docking score and rmsd
                        # for all isomers and conformers in mol
                        score = score_combine(min_dock / dock_iso_min, min_rmsd / rmsd_iso_min, 100)
                    else:
                        assert_never()

                    if score < best_iso_score:
                        best_iso_score = score
                        best_iso = isomer

                else:
                    # remove unsuitable isomers
                    if isomer.scores is None:
                        self.logger.info(
                            "For isomer %s in molecule %s no docking scores found!",
                            isomer,
                            mol,
                        )
                    mol.remove_isomer(isomer)

            # remove all superflous isomers
            for isomer in mol.molecules[:]:
                if isomer != best_iso:
                    mol.remove_isomer(isomer)

        self.out.send(mols)


@pytest.fixture
def path_ref(shared_datadir: Any) -> Any:
    return shared_datadir / "rmsd-filter-ref.sdf"


@pytest.fixture
def iso_paths(shared_datadir: Any) -> Any:
    return [shared_datadir / "rmsd-filter-iso1.sdf", shared_datadir / "rmsd-filter-iso2.sdf"]


def test_RMSD_Filter(path_ref: Any, iso_paths: Any) -> None:
    """Test RMSD_Filter"""

    iso_list = [Isomer.from_sdf(path, read_conformers=True) for path in iso_paths]
    for iso in iso_list:
        iso.score_tag = "energy"

    rig = TestRig(RMSDFilter)
    params = {
        "ref_lig": Path(path_ref),
        "reference_charge_type": "ref",
        "strict_target_charge": False,
    }
    res = rig.setup_run(parameters=params, inputs={"inp": [[IsomerCollection(iso_list)]]})
    filtered = res["out"].get()

    assert filtered is not None
    assert filtered[0].molecules[0].scored
    assert filtered[0].molecules[0].n_conformers == 1
    assert len(filtered[0].molecules) == 1
