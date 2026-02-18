"""Nodes for tagging isomers based on molecular properties"""

import ast
import fnmatch
import itertools
import math
from pathlib import Path
from random import shuffle
import operator as op
from typing import Any, Callable, Literal, Sequence, cast

import numpy as np
from numpy.typing import NDArray
import pytest

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Flag, ParameterException
from maize.utilities.chem import IsomerCollection, Isomer, rmsd as chemrmsd
from maize.utilities.testing import TestRig


_ValidOpsType = Literal["min", "max", "mean", "std", "var", "first", "last"]

SCORE_AGG_RULES: tuple[tuple[str, Literal["min", "max"]], ...] = (
    ("docking", "min"),
    ("minimizedAffinity", "min"),
    ("CNN", "max"),
    ("num_", "max"),
    ("HBAcceptor", "max"),
    ("HBDonor", "max"),
    ("Cationic", "max"),
    ("PiCation", "max"),
    ("PiStacking", "max"),
)


def infer_agg_from_tag(name: str) -> Literal["min", "max"]:
    """Infer 'min' or 'max' aggregation mode from a score/tag name based on SCORE_AGG_RULES."""
    lname = name.lower()
    for pattern, agg in SCORE_AGG_RULES:
        if pattern.lower() in lname:
            return agg
    return "min"


class TagAgg(Node):
    """Set new tags by performing aggregation over tags"""

    tags = {"tagger", "aggregation", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    expressions: Parameter[dict[str, tuple[str | list[str], _ValidOpsType]]] = Parameter()
    """
    Dictionary with keys as new target tags and values representing
    both the tags to operate on and the operation to be performed.
    The tags can be listed explicitly or given as a wildcard. Example:

    .. code-block:: python

       node.expressions.set({
           "score": (["score-1", "score-2"], "mean"),
           "agg": ("*deltaG", "min"),
       })

    Here, two new tags will be created, with the first one ('score')
    being set to the mean of 'score-1' and 'score-2' and the second
    one ('agg') being set to the result of taking the minimum between
    all tags ending in 'deltaG'

    Allowed aggregation functions are 'min', 'max', 'mean'

    """

    AGG: dict[_ValidOpsType, Callable[[Sequence[Any]], float]] = {
        "min": np.nanmax,
        "max": np.nanmax,
        "mean": np.nanmean,
        "var": np.nanvar,
        "std": np.nanstd,
        "first": lambda arr: arr[0],
        "last": lambda arr: arr[-1],
    }

    @staticmethod
    def _expand_tags(tags: list[str] | str, all_tags: dict[str, float | int]) -> list[float | int]:
        if isinstance(tags, str):
            tags = [tags]
        expanded = itertools.chain(*(fnmatch.filter(all_tags.keys(), tag) for tag in tags))
        return [all_tags[tag] for tag in expanded]

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                for new_tag, (tags, agg) in self.expressions.value.items():
                    values = self._expand_tags(tags, iso.tags)
                    iso.set_tag(new_tag, self.AGG[agg](values))
        self.out.send(mols)


_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
}

_UNOPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: op.neg,
}

_FUNCS: dict[str, Callable[[Any], Any]] = {
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
}


def _eval(node: Any, iso: Isomer) -> Any:
    match node:
        case ast.Call(func, args, _):
            return _eval(func, iso)(*(_eval(arg, iso) for arg in args))
        case ast.Constant(value):
            return value
        case ast.Name(value) if value in iso.tags:
            return iso.tags[value]
        case ast.Name(value) if value in _FUNCS:
            return _FUNCS[value]
        case ast.BinOp(left, op, right):
            return _BINOPS[type(op)](_eval(left, iso), _eval(right, iso))
        case ast.UnaryOp(op, operand):
            return _UNOPS[type(op)](_eval(operand, iso))
        case _:
            raise TypeError(node)


class TagMath(Node):
    """Set new tags by performing arithmetic operations on other tags"""

    tags = {"tagger", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    expressions: Parameter[dict[str, str]] = Parameter()
    """
    Dictionary of expressions with keys as the (new) target tags,
    and values as arithmetic expressions. These expressions can use
    existing tag names and mathematical operators. Example:

    .. code-block:: python

       node.expressions.set({
           "agg": "CNNaffinity / rmsd",
       })

    Here, two new tags will be created, with the first one ('score')
    being set to the mean of all tags ending in 'minimizedAffinity',
    and the second one ('agg') being set to the result of dividing
    the CNNaffinity tag by the rmsd tag.

    Allowed operators are '+', '-', '*', '/', '**'
    Allowed functions are 'exp', 'sqrt', 'log', 'log10'

    """

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                for new_tag, expr in self.expressions.value.items():
                    value = _eval(ast.parse(expr, mode="eval").body, iso)
                    iso.set_tag(new_tag, value)
        self.out.send(mols)


class RenameTag(Node):
    """Rename tags in a molecule"""

    tags = {"tagger", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    mapping: Parameter[dict[str, str]] = Parameter()
    """Key-value pairs for renaming"""

    overwrite: Flag = Flag(default=False)
    """Whether to overwrite existing tags"""

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                for old_name, new_name in self.mapping.value.items():
                    if not iso.has_tag(old_name):
                        self.logger.warning("Tag with name '%s' not found, skipping...", old_name)
                        continue
                    if iso.has_tag(new_name) and not self.overwrite.value:
                        self.logger.warning(
                            "Tag with new name '%s' already exists in molecule, "
                            "use 'overwrite' to enable overwriting of the tag, "
                            "skipping for now...",
                            new_name,
                        )
                        continue
                    iso.set_tag(new_name, iso.get_tag(old_name))
        self.out.send(mols)


class SetTag(Node):
    """Tag each molecule with a constant"""

    tags = {"tagger", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    values: Parameter[dict[str, Any]] = Parameter()
    """Key-value pairs for tagging"""

    score: Flag = Flag(default=False)
    """Whether to set the tag as a score"""

    score_agg: Parameter[Literal["min", "max"]] = Parameter(default="min")
    """The type of aggregation to perform when setting the tag as a score"""

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                for tag, value in self.values.value.items():
                    iso.set_tag(tag, value)
                    if self.score.value:
                        iso.add_score_tag(tag)
                        iso.primary_score_tag = tag
                        iso.set_tag("origin", self.name)
        self.out.send(mols)


class RemoveTags(Node):
    """Remove tags in a molecule"""

    tags = {"tagger", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    tags_to_remove: Parameter[list[str]] = Parameter(default_factory=list)
    """Tags to remove from each molecule"""

    tags_to_keep: Parameter[list[str]] = Parameter(default_factory=list)
    """Tags to keep in each molecule"""

    def prepare(self) -> None:
        if bool(self.tags_to_keep.value) == bool(self.tags_to_remove.value):
            raise ParameterException("Only one of 'tags_to_keep' or 'tags_to_remove' can be provided")

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                if self.tags_to_keep.value:
                    for tag in iso.tags:
                        if tag not in self.tags_to_keep.value:
                            iso.remove_tag(tag)
                elif self.tags_to_remove.value:
                    for tag in self.tags_to_remove.value:
                        if iso.has_tag(tag):
                            iso.remove_tag(tag)

        self.out.send(mols)


class FilterScoreTag(Node):
    """Only retain desired score tags in a molecule"""

    tags = {"tagger", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    scores_to_keep: Parameter[list[str] | str] = Parameter(optional=True)
    """Score Tags to keep in each molecule"""

    def run(self) -> None:
        mols = self.inp.receive()

        if self.scores_to_keep.is_set:
            keep = self.scores_to_keep.value
            if isinstance(keep, str):
                keep = [keep]
            for mol in mols:
                for iso in mol.molecules:
                    for score_name in iso._score_tags:
                        _, _, name = score_name.split("__")
                        if name not in keep:
                            iso.remove_tag(score_name)

        self.out.send(mols)


class SetName(Node):
    """Set the name of each molecule based on a tag"""

    tags = {"chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    tag: Parameter[str] = Parameter()
    """Tag to use for naming"""

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                if not iso.has_tag(self.tag.value):
                    self.logger.warning(
                        "Tag %s not found in isomer %s", self.tag.value, iso.name or iso.inchi
                    )
                    continue
                iso.name = str(iso.get_tag(self.tag.value))
        self.out.send(mols)


INCHIKEY_FIELD_NAME = "InChIKey"


class TagWithInchiKey(Node):
    """Add InchiKey as tag"""

    tags = {"tagger", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    def run(self) -> None:
        mols = self.inp.receive()

        for mol in mols:
            inchi_key = mol.complete_inchi_key
            mol.set_tag(INCHIKEY_FIELD_NAME, inchi_key)

            for iso in mol.molecules:
                iso.set_tag(INCHIKEY_FIELD_NAME, inchi_key)

        self.out.send(mols)


class TagIndex(Node):
    """Tag each molecule with it's index in the list to allow sorting and re-merging operations"""

    tags = {"tagger", "chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with index tags"""

    tag: Parameter[str] = Parameter(default="idx")
    """The tag to use for indexing"""

    def run(self) -> None:
        mols = self.inp.receive()
        for i, mol in enumerate(mols):
            mol.set_tag(self.tag.value, i)
        self.out.send(mols)


class LogTags(Node):
    """Log the value of a tag for a set of molecules"""

    tags = {"chemistry", "saving"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules"""

    tag: Parameter[str] = Parameter()
    """The tag to use for logging"""

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            val = mol.get_tag(self.tag.value, default="")
            self.logger.info("Molecule '%s', %s = %s", mol.smiles, self.tag.value, val)
            for iso in mol.molecules:
                val = iso.get_tag(self.tag.value, default="")
                self.logger.info("Isomer '%s', %s = %s", iso.inchi, self.tag.value, val)
                for i, conf in enumerate(iso.conformers):
                    val = conf.get_tag(self.tag.value, default="")
                    self.logger.info("Conformer %s, %s = %s", i, self.tag.value, val)
        self.out.send(mols)


SortableRDKitTagType = bool | int | float | str


class SortByTag(Node):
    """Sort a list of `IsomerCollection` based on a tag"""

    tags = {"chemistry"}

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to be sorted"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule list output"""

    tag: Parameter[str] = Parameter(default="idx")
    """The tag to use for sorting"""

    reverse: Flag = Flag(default=False)
    """Whether to use reverse order"""

    def run(self) -> None:
        mols = self.inp.receive()
        mols.sort(
            key=lambda mol: cast(SortableRDKitTagType, mol.get_tag(self.tag.value)),
            reverse=self.reverse.value,
        )
        self.out.send(mols)


class SetPrimaryScore(Node):
    """Sets the tag used to score / rank molecules"""

    tags = {"chemistry", "scoring"}

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to be sorted"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule list output"""

    tag: Parameter[str] = Parameter()
    """The name of the score to set as primary"""

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                if self.tag.value not in iso.scores:
                    self.logger.warning(
                        "Score '%s' not found in '%s', skipping...",
                        self.tag.value,
                        iso.name or iso.inchi,
                    )
                    continue
                iso.primary_score_tag = self.tag.value
            mol.primary_score_tag = self.tag.value
        self.out.send(mols)


class SetScoreTag(Node):
    """Sets the tag used to score / rank molecules"""

    tags = {"chemistry", "scoring"}

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to be sorted"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule list output"""

    tag: Parameter[str] = Parameter(optional=True)
    """The tag to use for sorting, will use the existing score tag if not set"""

    agg: Parameter[Literal["min", "max"]] = Parameter(default="min")
    """How to aggregate values across isomers"""

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                if self.tag.is_set:
                    iso.add_score_tag(self.tag.value, self.agg.value)
        self.out.send(mols)


class ExtractTag(Node):
    """
    Extract a specific numeric tag from molecules. The output is guaranteed
    to have the same length and ordering as the input molecules.

    """

    tags = {"chemistry", "utility"}

    AGGREGATORS: dict[str, Callable[[Sequence[Any]], float]] = {
        "min": min,
        "max": max,
        "mean": np.mean,
        "first": lambda arr: arr[0],
        "last": lambda arr: arr[-1],
    }

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to be sorted"""

    out: Output[NDArray[np.float32]] = Output()
    """Tag output"""

    tag: Parameter[str] = Parameter()
    """The tag to use for sorting"""

    agg: Parameter[Literal["min", "max", "mean", "first", "last"]] = Parameter(default="mean")
    """How to aggregate values across isomers"""

    def run(self) -> None:
        mols = self.inp.receive()
        agg = self.AGGREGATORS[self.agg.value]
        key = self.tag.value
        outputs: list[float] = []
        for mol in mols:
            vals = [
                float(cast(SortableRDKitTagType, iso.get_tag(key)))
                for iso in mol.molecules
                if iso.has_tag(key)
            ]
            outputs.append(agg(vals) if vals else np.nan)

        self.out.send(np.array(outputs))


class PadScoreTag(Node):
    """
    Pad missing tags provided in padtag with value NaN with options to infer aggregation method and set as tag.
    """

    tags = {"chemistry", "utility"}

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules to be checked"""

    out: Output[list[IsomerCollection]] = Output()
    """Output"""

    padtag: Parameter[list[str] | str] = Parameter(optional=True)
    """Tags to pad"""

    infer: Flag = Flag(default=True)
    """Whether to infer aggregation method from tag name."""

    settag: Flag = Flag(default=False)
    """Whether to set a tag"""

    score_agg: Parameter[Literal["min", "max"]] = Parameter(default="min")
    """The type of aggregation to perform when setting the tag as a score"""

    def run(self) -> None:
        mols = self.inp.receive()
        infer = self.infer.value

        if not self.padtag.is_set:
            self.out.send(mols)
        else:
            padtag_names = self.padtag.value
            if isinstance(padtag_names, str):
                padtag_names = [padtag_names]
            for mol in mols:
                for iso in mol.molecules:
                    tag_list = iso.scores.keys()
                    for name in padtag_names:
                        if name not in tag_list:
                            if infer:
                                agg = infer_agg_from_tag(name)
                            else:
                                agg = self.score_agg.value
                            iso.add_score(name, np.nan, agg)
                            if self.settag.value:
                                iso.set_tag(name, np.nan)

        self.out.send(mols)


class ExtractScores(Node):
    """Extract scores from molecules"""

    tags = {"chemistry", "utility"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: Output[NDArray[np.float32]] = Output()
    """List of molecules with RMSD tags"""

    def run(self) -> None:
        mols = self.inp.receive()
        scores = np.array([mol.primary_score for mol in mols])
        self.out.send(scores)


class MergeScoreTag(Node):
    """Copy scores from matching isomers in library 2 to library 1. Send output as input if not tags were specified."""

    tags = {"chemistry", "utility"}

    inp1: Input[list[IsomerCollection]] = Input()
    """First list of molecules (target: scores will be added here)."""

    inp2: Input[list[IsomerCollection]] = Input()
    """Second list of molecules (source of scores)."""

    tags: Parameter[list[str] | str] = Parameter(optional=True)
    """Names of scores to copy."""

    moltag1: Parameter[str] = Parameter(default="m_molid")
    """Tag in library 1 to match molecules."""

    isotag1: Parameter[str] = Parameter(default="m_isoid")
    """Tag in library 1 to match isomers."""

    moltag2: Parameter[str] = Parameter(default="m_molid")
    """Tag in library 2 to match molecules."""

    isotag2: Parameter[str] = Parameter(default="m_isoid")
    """Tag in library 2 to match isomers."""

    infer: Flag = Flag(default=True)
    """Whether to infer aggregation method from score name."""

    score_agg: Parameter[Literal["min", "max"]] = Parameter(default="min")
    """The type of aggregation to perform when setting the tag as a score"""

    out: Output[list[IsomerCollection]] = Output()
    """First list of molecules with scores added."""

    def run(self) -> None:
        mols1 = self.inp1.receive()
        mols2 = self.inp2.receive()
        moltag1 = self.moltag1.value
        isotag1 = self.isotag1.value
        moltag2 = self.moltag2.value
        isotag2 = self.isotag2.value
        infer = self.infer.value

        # If no tags requested, just pass through
        if not self.tags.is_set:
            self.out.send(mols1)
            self.logger.warning("No score tags specified, passing input unchanged.")
            return

        score_names = self.tags.value
        if isinstance(score_names, str):
            score_names = [score_names]

        for mol1 in mols1:
            ref_mol_tag = mol1.get_tag(moltag1)
            for iso1 in mol1.molecules:
                ref_iso_tag = iso1.get_tag(isotag1)
                had_match = False  # track per iso1

                for mol2 in mols2:
                    mol2_tag = mol2.get_tag(moltag2)
                    if mol2_tag != ref_mol_tag:
                        continue

                    for iso2 in mol2.molecules:
                        iso2_tag = iso2.get_tag(isotag2)
                        if iso2_tag != ref_iso_tag:
                            continue

                        had_match = True
                        # Copy scores from iso2
                        for name in score_names:
                            if name not in iso2.scores:
                                continue
                            score = iso2.scores[name]
                            if infer:
                                agg = iso2.score_agg.get(name, infer_agg_from_tag(name))
                            else:
                                agg = self.score_agg.value
                            iso1.add_score(name, score, agg=agg)

                # If no matching (mol2, iso2) was found
                if not had_match:
                    for name in score_names:
                        if infer:
                            agg = infer_agg_from_tag(name)
                        else:
                            agg = self.score_agg.value
                        iso1.add_score(name, np.nan, agg=agg)

        self.out.send(mols1)


class RMSD(Node):
    """Calculates RMSDs to a reference molecule"""

    tags = {"chemistry", "scorer", "tagger"}

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    inp_ref: Input[Isomer] = Input(cached=True)
    """Reference isomer to compute RMSD to"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with RMSD tags"""

    timeout: Parameter[int] = Parameter(default=1)
    """Time to wait for the RMSD calculation before returning NaN"""

    def run(self) -> None:
        mols = self.inp.receive()
        ref = self.inp_ref.receive()
        self.logger.info("Using '%s' as reference", ref.inchi)
        timeout = self.timeout.value
        for mol in mols:
            if not mol.molecules:
                mol.set_tag("rmsd", np.inf)
            for iso in mol.molecules:
                rmsds = chemrmsd(iso, ref, timeout=timeout)
                if rmsds is None:
                    continue

                for conf, rmsd in zip(iso.conformers, rmsds):
                    conf.set_tag("rmsd", rmsd)
                    conf.add_score("rmsd", rmsd, agg="min")

                iso.set_tag("rmsd", min(rmsds))
                iso.add_score("rmsd", rmsds, agg="min")
                self.logger.info("Isomer '%s' RMSD %s", iso.name or iso.inchi, min(rmsds))
        self.out.send(mols)


@pytest.fixture
def tagged_mols() -> list[IsomerCollection]:
    mols = [
        IsomerCollection.from_smiles("CCC"),
        IsomerCollection.from_smiles("CCCC"),
    ]
    for i, mol in enumerate(mols):
        mol.embed()
        for iso in mol.molecules:
            iso.set_tag("score", -i - 0.5)
            iso.set_tag("foo-score", -i - 2.5)
            iso.add_score_tag("score")
    return mols


@pytest.fixture
def indexed_mols() -> list[IsomerCollection]:
    mols = [
        IsomerCollection.from_smiles("CC"),
        IsomerCollection.from_smiles("CCC"),
        IsomerCollection.from_smiles("CCCC"),
    ]
    for i, mol in enumerate(mols):
        mol.embed()
        for iso in mol.molecules:
            iso.set_tag("idx", i)
    return mols


@pytest.fixture
def lib_a() -> list[IsomerCollection]:
    mols = [
        IsomerCollection.from_smiles("CC"),
        IsomerCollection.from_smiles("CCC"),
    ]
    for i, mol in enumerate(mols):
        mol.embed()
        mol.set_tag("m_molid", str(i))
        for j, iso in enumerate(mol.molecules):
            iso.set_tag("m_isoid", str(j))
            iso.add_score("CNNscore", 0.3, agg="max")
    return mols


@pytest.fixture
def lib_b() -> list[IsomerCollection]:
    mols = [
        IsomerCollection.from_smiles("CC"),
    ]
    for i, mol in enumerate(mols):
        mol.embed()
        mol.set_tag("m_molid", str(i))
        for j, iso in enumerate(mol.molecules):
            iso.set_tag("m_isoid", str(j))
            iso.add_score("num_interaction", 2, agg="max")
    return mols


@pytest.fixture
def path_ref(shared_datadir: Path) -> Path:
    return shared_datadir / "rmsd-filter-ref.sdf"


@pytest.fixture
def iso_paths(shared_datadir: Path) -> list[Path]:
    return [shared_datadir / "rmsd-filter-iso1.sdf", shared_datadir / "rmsd-filter-iso2.sdf"]


class TestSuiteTaggers:
    def test_TagAgg(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(TagAgg)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]},
            parameters={
                "expressions": {
                    "new-score": ("*score", "min"),
                    "other-score": (["score", "foo-score"], "mean"),
                }
            },
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert np.allclose(mols[0].molecules[0].get_tag("new-score"), -2.5, atol=0.01)
        assert np.allclose(mols[1].molecules[0].get_tag("new-score"), -3.5, atol=0.01)
        assert np.allclose(mols[0].molecules[0].get_tag("other-score"), -1.5, atol=0.01)
        assert np.allclose(mols[1].molecules[0].get_tag("other-score"), -2.5, atol=0.01)

    def test_TagMath(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(TagMath)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]},
            parameters={
                "expressions": {
                    "new-score": "exp(score) * 0.5 - 2.0",
                    "other-score": "sqrt(2) * score + 17 ** 2",
                }
            },
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert np.allclose(mols[0].molecules[0].get_tag("new-score"), -1.6967, atol=0.01)
        assert np.allclose(mols[1].molecules[0].get_tag("new-score"), -1.8884, atol=0.01)
        assert np.allclose(mols[0].molecules[0].get_tag("other-score"), 288.2928, atol=0.01)
        assert np.allclose(mols[1].molecules[0].get_tag("other-score"), 286.8786, atol=0.01)

    def test_RemoveTags(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(RemoveTags)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]}, parameters={"tags_to_keep": ["foo-score"]}
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].has_tag("foo-score")
        assert not mols[1].molecules[0].has_tag("score")

        res = rig.setup_run(inputs={"inp": [tagged_mols]}, parameters={"tags_to_remove": ["score"]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].has_tag("foo-score")
        assert not mols[1].molecules[0].has_tag("score")

    def test_PadScoreTag(self, lib_a: list[IsomerCollection]) -> None:
        rig = TestRig(PadScoreTag)
        res = rig.setup_run(
            inputs={"inp": [lib_a]},
            parameters={"padtag": ["CNNaffinity", "foo"], "infer": True},
        )
        mols = res["out"].get()

        assert mols is not None
        assert len(mols) == 2
        assert np.isnan(mols[0].molecules[0].scores["foo"])
        assert mols[0].molecules[0].score_agg["foo"] == "min"
        assert np.isnan(mols[0].molecules[0].scores["CNNaffinity"])
        assert mols[0].molecules[0].score_agg["CNNaffinity"] == "max"

    def test_MergeScoreTag(
        self, lib_a: list[IsomerCollection], lib_b: list[IsomerCollection]
    ) -> None:
        rig = TestRig(MergeScoreTag)
        res = rig.setup_run(
            inputs={"inp1": [lib_a], "inp2": [lib_b]}, parameters={"tags": ["num_interaction"]}
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert np.isnan(mols[1].molecules[0].scores["num_interaction"])
        assert (
            mols[0].molecules[0].scores.items() == {"CNNscore": 0.3, "num_interaction": 2}.items()
        )
        assert mols[0].molecules[0].score_agg["num_interaction"] == "max"

    def test_RemoveTags_bad(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(RemoveTags)
        with pytest.raises(ParameterException):
            rig.setup_run(
                inputs={"inp": [tagged_mols]},
                parameters={"tags_to_keep": ["foo-score"], "tags_to_remove": ["score"]},
            )

    def test_RenameTag(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(RenameTag)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]},
            parameters={"mapping": {"foo-score": "foo-new", "score": "foo-score"}},
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].get_tag("foo-new") == -2.5
        assert mols[1].molecules[0].get_tag("foo-new") == -3.5
        assert mols[0].molecules[0].get_tag("foo-score") == -2.5
        assert mols[1].molecules[0].get_tag("foo-score") == -3.5

    def test_RenameTag_overwrite(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(RenameTag)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]},
            parameters={
                "mapping": {"foo-score": "foo-new", "score": "foo-score"},
                "overwrite": True,
            },
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].get_tag("foo-new") == -2.5
        assert mols[1].molecules[0].get_tag("foo-new") == -3.5
        assert mols[0].molecules[0].get_tag("foo-score") == -0.5
        assert mols[1].molecules[0].get_tag("foo-score") == -1.5

    def test_SetTag(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(SetTag)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]}, parameters={"values": {"foo": 5.0, "bar": "baz"}}
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].get_tag("foo") == 5.0
        assert mols[1].molecules[0].get_tag("foo") == 5.0
        assert mols[0].molecules[0].get_tag("bar") == "baz"
        assert mols[1].molecules[0].get_tag("bar") == "baz"

    def test_SetTag_score(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(SetTag)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]}, parameters={"values": {"foo": 5.0}, "score": True}
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].get_tag("foo") == 5.0
        assert mols[1].molecules[0].get_tag("foo") == 5.0
        assert mols[0].molecules[0].scores["foo"] == 5.0
        assert mols[1].molecules[0].scores["foo"] == 5.0
        assert mols[0].primary_score == 5.0
        assert mols[1].primary_score == 5.0

    def test_SetName(self, tagged_mols: list[IsomerCollection]) -> None:
        i = 0
        for mol in tagged_mols:
            for iso in mol.molecules:
                iso.set_tag("foo", f"AZ{i}")
                i += 1
        rig = TestRig(SetName)
        res = rig.setup_run(inputs={"inp": [tagged_mols]}, parameters={"tag": "foo"})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].name == "AZ0"
        assert mols[1].molecules[0].name == "AZ1"

    def test_TagIndex(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(TagIndex)
        res = rig.setup_run(inputs={"inp": [tagged_mols]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 2
        assert mols[0].molecules[0].get_tag("idx") == 0
        assert mols[1].molecules[0].get_tag("idx") == 1

    def test_SortMolecules(self, indexed_mols: list[IsomerCollection]) -> None:
        rig = TestRig(SortByTag)
        shuffle(indexed_mols)
        res = rig.setup_run(inputs={"inp": [indexed_mols]})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3
        for i in range(3):
            assert mols[i].molecules[0].get_tag("idx") == i

    def test_SortMolecules_reverse(self, indexed_mols: list[IsomerCollection]) -> None:
        rig = TestRig(SortByTag)
        shuffle(indexed_mols)
        res = rig.setup_run(inputs={"inp": [indexed_mols]}, parameters={"reverse": True})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3
        for i in range(3):
            assert mols[i].molecules[0].get_tag("idx") == 2 - i

    def test_SetScoreTag(self, indexed_mols: list[IsomerCollection]) -> None:
        rig = TestRig(SetScoreTag)
        res = rig.setup_run(inputs={"inp": [indexed_mols]}, parameters={"tag": "idx"})
        mols = res["out"].get()
        assert mols is not None
        assert mols[0].primary_score == 0
        assert mols[0].molecules[0].primary_score_tag == "idx"

    def test_ExtractTag(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(ExtractScores)
        res = rig.setup_run(
            inputs={"inp": [tagged_mols]}, parameters={"tag": "score", "agg": "mean"}
        )
        scores = res["out"].get()
        assert scores is not None
        assert np.allclose(scores, [-0.5, -1.5])

    def test_ExtractScores(self, tagged_mols: list[IsomerCollection]) -> None:
        rig = TestRig(ExtractScores)
        res = rig.setup_run(inputs={"inp": [tagged_mols]})
        scores = res["out"].get()
        assert scores is not None
        assert np.allclose(scores, [-0.5, -1.5])

    def test_RMSD(self, path_ref: Path, iso_paths: list[Path]) -> None:
        iso_list = [Isomer.from_sdf(path) for path in iso_paths]
        iso_list[1].embed(4)
        ref = Isomer.from_sdf(path_ref)

        rig = TestRig(RMSD)
        res = rig.setup_run(inputs={"inp": [[IsomerCollection(iso_list)]], "inp_ref": [ref]})
        tagged = res["out"].get()

        assert tagged is not None
        assert np.allclose(tagged[0].molecules[0].get_tag("rmsd"), 3.36, 0.01)
        assert np.allclose(tagged[0].molecules[1].get_tag("rmsd"), 3.75, 0.01)
        assert np.allclose(tagged[0].molecules[0].scores["rmsd"], 3.36, 0.01)
        assert np.allclose(tagged[0].molecules[1].primary_score, 3.75, 0.01)
        assert np.allclose(tagged[0].molecules[1].conformers[0].get_tag("rmsd"), 3.75, 0.01)
        assert np.allclose(tagged[0].molecules[1].conformers[1].get_tag("rmsd"), 3.75, 0.01)
