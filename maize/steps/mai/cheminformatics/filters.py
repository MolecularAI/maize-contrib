"""Nodes for filtering down isomer collections"""

import copy
import functools
from pathlib import Path
from typing import Any, Callable, List, Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Flag
from maize.utilities.chem import IsomerCollection, Isomer
from maize.utilities.chem.chem import Conformer
from maize.utilities.testing import TestRig
from maize.utilities.io import Config


class BestIsomerFilter(Node):
    """
    Filter a list of `IsomerCollection` to retain only the best
    compound according to their score or a user-defined tag.

    """
    tags = {"filter", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules as isomer collections after filters"""

    score_tag: Parameter[str] = Parameter(optional=True)
    """
    Tag or score to use for ranking. Will first look for a score under this name,
    and if not found look for a tag. Will use the primary score if not given.

    """

    descending: Flag = Flag(default=True)
    """
    Sort tag descending (lower = better), this setting
    will be ignored when using a score for sorting

    """

    @staticmethod
    def sorter(iso: Isomer, tag: str | None = None, desc: bool = True) -> float:
        INVALID = np.inf if desc else -np.inf
        if tag is None:
            if iso.primary_score_tag is None:
                return INVALID
            tag = iso.primary_score_tag
        if tag in iso.scores:
            return INVALID if np.isnan(iso.scores[tag]) else iso.scores[tag]
        if iso.has_tag(tag):
            return float(cast(float, iso.get_tag(tag, default=INVALID)))
        return INVALID

    def run(self) -> None:
        mols = self.inp.receive()
        tag = self.score_tag.value if self.score_tag.is_set else None

        # Get aggregation order, default is lower is better ("min")
        desc = self.descending.value
        for mol in mols:
            for iso in mol.molecules:
                if tag is not None and tag in iso.score_agg:
                    desc = iso.score_agg[tag] == "min"
                    break

        sorter = functools.partial(self.sorter, tag=tag, desc=desc)
        isomers = [sorted(mol.molecules, key=sorter, reverse=not desc) for mol in mols]
        new_mols = [
            IsomerCollection([isos[0]]) if isos else IsomerCollection([]) for isos in isomers
        ]
        self.out.send(new_mols)


class BestConformerFilter(Node):
    """
    Filter a list of `IsomerCollection` to retain only the best
    conformer according to their score or a user-defined tag.

    """
    tags = {"filter", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules as isomer collections after filters"""

    score_tag: Parameter[str] = Parameter(optional=True)
    """
    Tag or score to use for ranking. Will first look for a score under this name,
    and if not found look for a tag. Will use the primary score if not given.

    """

    descending: Flag = Flag(default=True)
    """
    Sort tag descending (lower = better), this setting
    will be ignored when using a score for sorting

    """

    @staticmethod
    def sorter(conf: Conformer, tag: str | None = None, desc: bool = True) -> float:
        INVALID = np.inf if desc else -np.inf
        if tag is None:
            if conf.primary_score_tag is None:
                return INVALID
            tag = conf.primary_score_tag
        if tag in conf.scores:
            return INVALID if np.isnan(conf.scores[tag]) else conf.scores[tag]
        if conf.has_tag(tag):
            return float(cast(float, conf.get_tag(tag, default=INVALID)))
        return INVALID

    def run(self) -> None:
        mols = self.inp.receive()
        tag = self.score_tag.value if self.score_tag.is_set else None

        # Get aggregation order, default is lower is better ("min")
        desc = self.descending.value
        for mol in mols:
            for iso in mol.molecules:
                if tag is not None and tag in iso.score_agg:
                    desc = iso.score_agg[tag] == "min"
                    break

        sorter = functools.partial(self.sorter, tag=tag, desc=desc)
        for mol in mols:
            for iso in mol.molecules:
                best = sorted(iso.conformers, key=sorter, reverse=not desc)[0]
                iso.clear_conformers()
                iso.add_conformer(best)

        self.out.send(mols)


# Originally: IsomerCollectionTagFilter
class TagFilter(Node):
    """
    Filter a list of `IsomerCollection` objects by their tags

    """
    tags = {"filter", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules as isomer collections after filtering"""

    must_have_tags: Parameter[list[str]] = Parameter(default_factory=list)
    """Tags that must be present in output, any value"""

    min_value_tags: Parameter[dict[str, Any]] = Parameter(default_factory=dict)
    """Tags whose numeric value must be >= to a minimum value"""

    max_value_tags: Parameter[dict[str, Any]] = Parameter(default_factory=dict)
    """Tags whose numeric value must be <= to a maximum value"""

    exact_value_tags: Parameter[dict[Any, Any]] = Parameter(default_factory=dict)
    """Tags whose value (any type) must correspond to the a provided key"""

    def run(self) -> None:
        isomer_collection_list = self.inp.receive()

        must_have_tags = self.must_have_tags.value
        min_value_tags = self.min_value_tags.value
        max_value_tags = self.max_value_tags.value
        exact_value_tags = self.exact_value_tags.value

        def _numeric_key_check(key: str, tag_dictionary: dict[str, Any]) -> bool:
            try:
                return not np.isnan(float(tag_dictionary[key]))
            except ValueError:
                return False

        def _generic_key_compare(
            key: str,
            value: Any,
            tag_dictionary: dict[str, Any],
            comparison: Callable[[Any, Any], bool] = lambda x, y: x > y,
        ) -> bool:
            return comparison(tag_dictionary[key], value)

        # this would otherwise modify the input in place
        isomer_collection_list_c = copy.deepcopy(isomer_collection_list)
        self.logger.info("entering filter with %i isomers" % len(isomer_collection_list_c))

        for ic_num, ic in enumerate(isomer_collection_list_c):
            remove_list = []
            for iso_num, isom in enumerate(ic.molecules):
                keep = True
                for tag in must_have_tags:
                    if not isom.has_tag(tag):
                        self.logger.debug(
                            "removing this iso #%i for mol %i for lack of key %s"
                            % (iso_num, ic_num, tag)
                        )
                        keep = False

                # get tags
                tag_dict = isom.tags

                # apply filters
                for tag_filter, enforce_numeric, comparison_op in zip(
                    (min_value_tags, max_value_tags, exact_value_tags),
                    (True, True, False),
                    (lambda x, y: x >= y, lambda x, y: x <= y, lambda x, y: x == y),
                ):
                    # filter
                    for key in tag_filter.keys():
                        if keep and enforce_numeric and not _numeric_key_check(key, tag_dict):
                            self.logger.debug(
                                "removing this iso #%i for mol %i for non-numeric key %s"
                                % (iso_num, ic_num, key)
                            )
                            keep = False
                        if keep and not _generic_key_compare(
                            key, tag_filter[key], tag_dict, comparison=comparison_op
                        ):
                            self.logger.debug(
                                "removing this iso #%i for mol %i, failing with value %s for key %s"
                                % (iso_num, ic_num, str(tag_dict[key]), key)
                            )
                            keep = False
                if not keep:
                    remove_list.append(isom)

            for isom in remove_list:
                ic.remove_isomer(isom)

        # clear out any empty isomer lists
        isomer_collection_list_c = [ic for ic in isomer_collection_list_c if ic.n_isomers > 0]
        self.logger.info("exiting filter with %i isomers" % len(isomer_collection_list_c))

        # send result
        self.out.send(isomer_collection_list_c)


# Originally: IsomerCollectionRankingFilter
class RankingFilter(Node):
    """
    Sorts a list of `IsomerCollection` objects by numeric
    tags and optionally filters to a max number.

    """
    tags = {"filter", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules as isomer collections after filtering"""

    tags_to_rank: Parameter[List[tuple[str, Literal["ascending", "descending"]]]] = Parameter()
    """List of tags (tag, ascending/descending") """

    max_output_length: Parameter[int] = Parameter(optional=True)
    """Return only the top *n* isomer collections after sorting"""

    def run(self) -> None:
        isomer_collection_list = self.inp.receive()

        sorting_instructions = self.tags_to_rank.value

        self.logger.info("entering filter with %i isomer collections" % len(isomer_collection_list))

        index_set = []
        for sorting_instruction in sorting_instructions:
            local_order = np.array(
                [
                    isoc.molecules[0].get_tag(sorting_instruction[0])
                    for isoc in isomer_collection_list
                ]
            )
            if sorting_instruction[1] == "descending":
                local_order *= -1
            index_set.append(local_order)

        final_sort = [
            isomer_collection_list[i]
            for i in sorted(
                range(len(isomer_collection_list)), key=list(zip(*index_set)).__getitem__
            )
        ]

        if self.max_output_length.is_set:
            final_sort = final_sort[0 : self.max_output_length.value]

        self.logger.info("exiting filter with %i isomer collections" % len(final_sort))

        # send result
        self.out.send(final_sort)


# Originally: IsomerFilter
class SMARTSFilter(Node):
    """
    Filter isomers according to occurring or missing
    substructures expressed as lists of SMARTS strings.

    """
    tags = {"filter", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules with isomers and conformations to filter"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with filtered isomers"""

    white_list_smarts: Parameter[list[str]] = Parameter(default_factory=list)
    """List of SMARTS that have to be in isomer to pass filter"""

    black_list_smarts: Parameter[list[str]] = Parameter(default_factory=list)
    """List of SMARTS that must not be in isomer to pass filter"""

    def run(self) -> None:
        mols = self.inp.receive()

        for mol in mols:
            n_before = mol.n_isomers
            # remove isomers that don't contain white list SMARTS
            for isomer in mol.molecules[:]:
                for smarts in self.white_list_smarts.value:
                    if not isomer.check_smarts(smarts):
                        mol.remove_isomer(isomer)
                        break

            # remove isomers that contain black list SMARTS
            for isomer in mol.molecules[:]:
                for smarts in self.black_list_smarts.value:
                    if isomer.check_smarts(smarts):
                        mol.remove_isomer(isomer)
                        break

            self.logger.info(f"Remaining isomers in {mol}: {mol.n_isomers}/{n_before}")

        self.out.send(mols)


T_arr_float = TypeVar("T_arr_float", NDArray[np.float32], float)


def score_combine(score1: T_arr_float, score2: T_arr_float, weight: float) -> T_arr_float:
    """Combines two normalized scores as geometric mean"""
    return cast(T_arr_float, np.sqrt(weight * np.square(score1 - 1) + np.square(score2 - 1)))


class ChargeFilter(Node):
    """Filter isomers by charge"""

    tags = {"filter", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules with isomers (from single SMILES) to filter"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with single isomer after filtering"""

    inp_ref: Input[Isomer] = Input(optional=True)
    """Reference ligand input, takes precedence over ``target_charge`` parameter"""

    target_charge: Parameter[list[int]] = Parameter(default_factory=list)
    """Only isomers with this total charge pass filter"""

    def run(self) -> None:
        mols = self.inp.receive()
        target_charge = self.target_charge.value
        if self.inp_ref.ready():
            ref = self.inp_ref.receive()
            target_charge = [ref.charge]

        for mol in mols:
            to_remove = [iso for iso in mol.molecules if iso.charge not in target_charge]
            for iso in to_remove:
                self.logger.info(
                    "Removing isomer '%s' with charge %s%s",
                    iso.name or iso.inchi,
                    "+" if iso.charge > 0 else "",
                    iso.charge,
                )
                mol.remove_isomer(iso)

        self.out.send(mols)


@pytest.fixture
def path_ref(shared_datadir: Path) -> Path:
    return shared_datadir / "rmsd-filter-ref.sdf"


@pytest.fixture
def iso_paths(shared_datadir: Path) -> list[Path]:
    return [shared_datadir / "rmsd-filter-iso1.sdf", shared_datadir / "rmsd-filter-iso2.sdf"]


class TestSuiteFilter:
    def test_BestIsomerFilter(self, test_config: Config) -> None:
        rig = TestRig(BestIsomerFilter, config=test_config)

        mols = [
            IsomerCollection([Isomer.from_smiles("C") for i in range(4)]),  # 1
            IsomerCollection([Isomer.from_smiles("C") for i in range(4)]),  # 2
        ]
        score_sets: list[dict[str, list[float | int]]] = [
            {"A": [1.0, 2.0, 3.0, 4.0], "B": [-1.0, np.nan, -3.0, -4.0]},
            {"A": [4.5, 3.5, 2.5, 1.5], "B": [-4.5, -3.5, -2.5, -1.5]},
        ]
        mol_counter = 0
        for mol, score_set in zip(mols, score_sets):
            mol_counter += 1
            for key in score_set.keys():
                iso_counter = 0
                for iso, prop_val in zip(mol.molecules, score_set[key]):
                    iso_counter += 1
                    iso.set_tag("iso_id", "-".join([str(mol_counter), str(iso_counter)]))
                    iso.set_tag(key, prop_val)
                    if key == "B":
                        iso.add_score(key, prop_val)
                        iso.add_score(key + "R", prop_val, agg="max")

        result = rig.setup_run(
            inputs={"inp": [mols]}, parameters={"score_tag": "A", "descending": False}
        )
        res = result["out"].get()
        assert res is not None
        assert [len(r.molecules) == 1 for r in res]
        assert res[0].molecules[0].get_tag("iso_id") == "1-4"
        assert res[1].molecules[0].get_tag("A") == 4.5

        # now test with score tag
        result = rig.setup_run(inputs={"inp": [mols]}, parameters={"descending": True})
        res = result["out"].get()
        assert res is not None
        assert [len(r.molecules) == 1 for r in res]
        assert res[0].molecules[0].get_tag("iso_id") == "1-4"
        assert res[0].molecules[0].get_tag("B") == -4.0
        assert res[1].molecules[0].get_tag("B") == -4.5

        result = rig.setup_run(inputs={"inp": [mols]}, parameters={"score_tag": "BR"})
        res = result["out"].get()
        assert res is not None
        assert [len(r.molecules) == 1 for r in res]
        assert res[0].molecules[0].get_tag("iso_id") == "1-1"
        assert res[0].molecules[0].scores["BR"] == -1.0
        assert res[1].molecules[0].scores["BR"] == -1.5

    def test_BestConformerFilter(self, test_config: Config, iso_paths: list[Path]) -> None:
        iso_list = [Isomer.from_sdf(path, read_conformers=True) for path in iso_paths]
        for iso in iso_list:
            scores = np.array([i for i, _ in enumerate(iso.conformers)], dtype=float)
            for i, conf in enumerate(iso.conformers):
                conf.set_tag("id", i)
            iso.add_score("score", scores)
            iso.add_score("score_r", scores, agg="max")

        rig = TestRig(BestConformerFilter, config=test_config)
        result = rig.setup_run(
            inputs={"inp": [[IsomerCollection(iso_list)]]}, parameters={"score_tag": "score"}
        )
        res = result["out"].get()
        assert res is not None
        mol = res[0]
        for iso in mol.molecules:
            assert iso.n_conformers == 1
            assert iso.conformers[0].scores == {"score": 0, "score_r": 0}
            assert iso.conformers[0].get_tag("id") == 0

    def test_BestConformerFilter_rev(self, test_config: Config, iso_paths: list[Path]) -> None:
        iso_list = [Isomer.from_sdf(path, read_conformers=True) for path in iso_paths]
        for iso in iso_list:
            scores = np.array([i for i, _ in enumerate(iso.conformers)], dtype=float)
            for i, conf in enumerate(iso.conformers):
                conf.set_tag("id", i)
            iso.add_score("score", scores)
            iso.add_score("score_r", scores, agg="max")

        rig = TestRig(BestConformerFilter, config=test_config)
        result = rig.setup_run(
            inputs={"inp": [[IsomerCollection(iso_list)]]}, parameters={"score_tag": "score_r"}
        )
        res = result["out"].get()
        assert res is not None
        mol = res[0]
        for iso in mol.molecules:
            assert iso.n_conformers == 1
            assert iso.conformers[0].scores == {"score": 19, "score_r": 19}
            assert iso.conformers[0].get_tag("id") == 19

    def test_TagFilter(self, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles("C"),  # 1
            IsomerCollection.from_smiles("CC"),  # 2
            IsomerCollection.from_smiles("CCCC"),  # 3
            IsomerCollection.from_smiles("CCCCC"),  # 4
            IsomerCollection.from_smiles("CCCCCC"),  # 5
            IsomerCollection.from_smiles("CCCCCCC"),  # 6
            IsomerCollection.from_smiles("CCCCCCCC"),  # 7
        ]
        tags_and_values: list[dict[str, str | int | float]] = [
            {"B": 1, "C": 1},  # missing tag A  1
            {"A": 1, "B": "foo", "C": "bar"},  # non-numeric tag B 2
            {"A": 1, "B": np.nan, "C": "bar"},  # non-numeric tag C 3
            {"A": 2, "B": 1, "C": "bar"},  # A is too high 4
            {"A": 1, "B": -1, "C": "bar"},  # B is too low 5
            {"A": 1, "B": 1, "C": "foo"},  # C is not bar 6
            {"A": 1, "B": 1, "C": "bar"},  # should pass 7
        ]
        counter = 0
        for mol, tag_dict in zip(mols, tags_and_values):
            counter += 1
            for iso in mol.molecules:
                for tag in tag_dict.keys():
                    iso.set_tag(tag, tag_dict[tag])
                    iso.name = f"molecule {counter}"

        rig = TestRig(TagFilter, config=test_config)
        result = rig.setup_run(
            inputs={"inp": [mols]},
            parameters={
                "must_have_tags": ["A", "B"],
                "min_value_tags": {"A": 1, "B": 1},
                "max_value_tags": {"A": 1, "B": 1},
                "exact_value_tags": {"C": "bar"},
            },
        )
        res = result["out"].get()
        assert res is not None
        assert len(res) == 1
        assert res[0].molecules[0].name == "molecule 7"

    def test_RankingFilter(self, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles("C"),  # 1
            IsomerCollection.from_smiles("CC"),  # 2
            IsomerCollection.from_smiles("CCCC"),  # 3
            IsomerCollection.from_smiles("CCCCC"),  # 4
            IsomerCollection.from_smiles("CCCCCC"),  # 5
            IsomerCollection.from_smiles("CCCCCCC"),  # 6
            IsomerCollection.from_smiles("CCCCCCCC"),  # 7
        ]
        tags_and_values = [
            {"A": 1, "B": -1},  # 1
            {"A": 2, "B": -2},  # 2
            {"A": 3, "B": -7},  # 3
            {"A": 3, "B": -6},  # 4
            {"A": 3, "B": -5},  # 5
            {"A": 2, "B": 4},  # 6
            {"A": 0, "B": 3},  # 7
        ]
        counter = 0
        for mol, tag_dict in zip(mols, tags_and_values):
            counter += 1
            for iso in mol.molecules:
                for tag in tag_dict.keys():
                    iso.set_tag(tag, tag_dict[tag])
                    iso.name = f"molecule {counter}"
        rig = TestRig(RankingFilter, config=test_config)
        rig = TestRig(RankingFilter, config=test_config)
        result = rig.setup_run(
            inputs={"inp": [mols]},
            parameters={
                "tags_to_rank": [("A", "descending"), ("B", "ascending")],
                "max_output_length": 3,
            },
        )
        res = result["out"].get()
        assert res is not None
        assert all([iso.molecules[0].get_tag("A") == 3 for iso in res])
        assert all([iso.molecules[0].get_tag("B") < -4 for iso in res])
        assert len(res) == 3

    def test_ChargeFilter(self, path_ref: Path, iso_paths: list[Path]) -> None:
        """Test ChargeFilter"""
        mol = IsomerCollection([Isomer.from_sdf(path, read_conformers=True) for path in iso_paths])

        rig = TestRig(ChargeFilter)
        res = rig.setup_run(inputs={"inp": [[mol]]}, parameters={"target_charge": [-2]})
        filtered = res["out"].get()
        assert filtered is not None
        assert filtered[0].n_isomers == 1
        assert filtered[0].molecules[0].charge == -2

        ref = Isomer.from_sdf(path_ref)
        rig = TestRig(ChargeFilter)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref]}, parameters={"target_charge": [-2]}
        )
        filtered = res["out"].get()
        assert filtered is not None
        assert filtered[0].n_isomers == 0
