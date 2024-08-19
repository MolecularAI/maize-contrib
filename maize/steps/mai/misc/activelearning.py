"""Active learning and associated utilities"""

import math
import pickle
import random
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, cast

import numpy as np
import pandas as pd
import pytest

from maize.core.node import Node, LoopedNode, NodeBuildException
from maize.core.graph import Graph
from maize.core.interface import (
    Input,
    Output,
    Parameter,
    Flag,
    FileParameter,
    Suffix,
)
from maize.steps.mai.cheminformatics.taggers import SortByTag, TagIndex
from maize.steps.plumbing import (
    Copy,
    Choice,
    Merge,
    Barrier,
    Multiplex,
    IntegerMap,
    IndexDistribute,
    TimeDistribute,
    CopyEveryNIter,
    MergeLists,
)
from maize.steps.io import FileBuffer, LoadData
from maize.steps.mai.misc import QptunaTrain, QptunaPredict, QptunaHyper
from maize.utilities.chem import IsomerCollection
from maize.utilities.testing import TestRig
from maize.utilities.utilities import deprecated


T = TypeVar("T")


SMILES_COLUMN = "smiles"


def _mols2csv(path: Path, mols: list[IsomerCollection]) -> None:
    new = pd.DataFrame({SMILES_COLUMN: [mol.smiles for mol in mols for _ in mol.molecules]})
    tags = {tag for mol in mols for iso in mol.molecules for tag in iso.tags}
    for tag in tags:
        new[tag] = [
            iso.get_tag(tag) if iso.has_tag(tag) else None for mol in mols for iso in mol.molecules
        ]
    new.to_csv(path)


class ActiveLearningProgressSingle(Node):
    """Save and log active learning progress for single runs"""

    inp: Input[list[IsomerCollection]] = Input()
    """Multiple list input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule list output"""

    save_epochs: Flag = Flag(default=False)
    """Whether to dump all molecules generated this epoch"""

    save_conformers: Flag = Flag(default=False)
    """Whether to dump all conformers generated this epoch"""

    save_location: FileParameter[Path] = FileParameter(exist_required=False, optional=True)
    """Directory to save epoch molecule dumps in"""

    def prepare(self) -> None:
        if (self.save_epochs.value or self.save_conformers.value) and not self.save_location.is_set:
            raise NodeBuildException(
                "You must specify the save location when "
                "using `save_epochs` or `save_conformers`"
            )

    def run(self) -> None:
        mols = self.inp.receive()
        self.logger.info("Received %s molecules", len(mols))
        if self.save_epochs.value or self.save_conformers.value:
            loc = self.save_location.filepath
            loc.mkdir(exist_ok=True)

            i = 0
            csv = loc / f"epoch-{i}.csv"
            while csv.exists():
                i += 1
                csv = loc / f"epoch-{i}.csv"

            _mols2csv(csv, mols)
            if self.save_conformers.value:
                folder = loc / f"epoch-{i}"
                folder.mkdir(exist_ok=True)
                for mol in mols:
                    for isomer in mol.molecules:
                        if isomer.get_tag("score_type", "") == "oracle":
                            isomer.to_sdf(folder / f"{isomer.inchi}.sdf", write_conformers=True)

        self.out.send(mols)
        scores = np.array([mol.primary_score for mol in mols])
        sorted_scores = sorted(scores)
        if all(
            iso.score_agg[iso.primary_score_tag] == "max"
            for mol in mols
            for iso in mol.molecules
            if iso.primary_score_tag is not None
        ):
            sorted_scores = sorted_scores[::-1]
        top10 = math.ceil(len(sorted_scores) * 0.1)

        msg = "Current statistics: \n"
        msg += f"   Median: {np.nanmedian(scores):4.4f}\n"
        msg += f"     Mean: {np.nanmean(scores):4.4f}\n"
        msg += f"      Std: {np.nanstd(scores):4.4f}\n"
        msg += f"  Top 10%: {np.nanmedian(sorted_scores[:top10]):4.4f}\n"
        self.logger.info(msg)


@deprecated("Please use single-iteration AL nodes instead")
class ActiveLearningProgress(LoopedNode):
    """Save and log active learning progress"""

    inp: Input[list[IsomerCollection]] = Input()
    """Multiple list input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule list output"""

    save_epochs: Flag = Flag(default=False)
    """Whether to dump all molecules generated this epoch"""

    save_conformers: Flag = Flag(default=False)
    """Whether to dump all conformers generated this epoch"""

    save_location: FileParameter[Path] = FileParameter(exist_required=False, optional=True)
    """Directory to save epoch molecule dumps in"""

    idx: int
    scores: list[np.floating[Any]]

    def prepare(self) -> None:
        if (self.save_epochs.value or self.save_conformers.value) and not self.save_location.is_set:
            raise NodeBuildException(
                "You must specify the save location when "
                "using `save_epochs` or `save_conformers`"
            )

        self.idx = 0
        self.scores = []

    def run(self) -> None:
        mols = self.inp.receive()
        self.logger.info("Received %s molecules", len(mols))
        if self.save_epochs.value or self.save_conformers.value:
            loc = self.save_location.filepath
            loc.mkdir(exist_ok=True)
            _mols2csv(loc / f"epoch-{self.idx}.csv", mols)
            if self.save_conformers.value:
                folder = loc / f"epoch-{self.idx}"
                folder.mkdir(exist_ok=True)
                for mol in mols:
                    for isomer in mol.molecules:
                        if isomer.get_tag("score_type", "") == "oracle":
                            isomer.to_sdf(folder / f"{isomer.inchi}.sdf", write_conformers=True)
        self.out.send(mols)
        self.idx += 1
        self.scores.append(np.nanmedian([mol.primary_score for mol in mols]))
        self.logger.info("Previous median scores: %s", self.scores)


class AcquisitionFunction(Node, register=False):
    """Acquisition function base"""

    inp: Input[list[IsomerCollection]] = Input()
    """Molecules with predicted scores"""

    out_oracle: Output[list[IsomerCollection]] = Output()
    """Output for the oracle"""

    out_surrogate: Output[list[IsomerCollection]] = Output()
    """Output for the surrogate / non-acquired molecules"""

    n_oracle: Parameter[int | float] = Parameter(default=32)
    """Number of molecules to send to the oracle, or the proportion of the total"""

    _n_oracle: int

    def _log_mol(self, mol: IsomerCollection) -> None:
        uncs = [
            float(cast(float, iso.get_tag("uncertainty")))
            for iso in mol.molecules
            if iso.has_tag("uncertainty")
        ]
        if uncs:
            self.logger.info(
                "Isomer '%s', score %s, uncertainty %s",
                mol.inchi,
                mol.primary_score,
                max(uncs),
            )
        else:
            self.logger.info("Isomer '%s', score %s", mol.inchi, mol.primary_score)

    def _recv(self) -> list[IsomerCollection]:
        mols = self.inp.receive()
        # Set the number of mols sent to the oracle if a proportion is given
        if isinstance(self.n_oracle.value, float):
            self._n_oracle = int(len(mols) * self.n_oracle.value)
        else:
            self._n_oracle = self.n_oracle.value
        for i, mol in enumerate(mols):
            mol.set_tag("idx", i)
        return mols

    def _send(
        self, oracle_mols: list[IsomerCollection], surrogate_mols: list[IsomerCollection]
    ) -> None:
        oracle_scores = np.array([mol.primary_score for mol in oracle_mols])
        self.logger.info(
            "Sending the following molecules to the oracle (mean: %s, median: %s)",
            np.nanmean(oracle_scores),
            np.nanmedian(oracle_scores),
        )
        for mol in oracle_mols:
            mol.set_tag("score_type", "oracle")
            self._log_mol(mol)
        self.out_oracle.send(oracle_mols)

        surrogate_scores = np.array([mol.primary_score for mol in surrogate_mols])
        self.logger.info(
            "Sending the following molecules to the surrogate (mean: %s, median: %s)",
            np.nanmean(surrogate_scores),
            np.nanmedian(surrogate_scores),
        )
        for mol in surrogate_mols:
            mol.set_tag("score_type", "surrogate")
            self._log_mol(mol)
        self.out_surrogate.send(surrogate_mols)


class Random(AcquisitionFunction):
    """Random acquisition function"""

    def run(self) -> None:
        mols = self._recv()
        random.shuffle(mols)
        oracle_mols = mols[: self._n_oracle]
        surrogate_mols = mols[self._n_oracle :]
        self._send(oracle_mols=oracle_mols, surrogate_mols=surrogate_mols)


class Greedy(AcquisitionFunction):
    """Greedy acquisition function"""

    def run(self) -> None:
        mols = self._recv()
        reverse = False
        if mols and (tag := mols[0].primary_score_tag):
            reverse = mols[0].score_agg[tag] == "max"
        mols.sort(key=lambda mol: mol.primary_score, reverse=reverse)
        oracle_mols = mols[: self._n_oracle]
        surrogate_mols = mols[self._n_oracle :]
        self._send(oracle_mols=oracle_mols, surrogate_mols=surrogate_mols)


class EpsilonGreedy(AcquisitionFunction):
    """Eps-Greedy acquisition function"""

    epsilon: Parameter[float] = Parameter(default=0.1)
    """Proportion of random rather than greedy selection"""

    def run(self) -> None:
        mols = self._recv()
        reverse = False
        if mols and (tag := mols[0].primary_score_tag):
            reverse = mols[0].score_agg[tag] == "max"
        mols.sort(key=lambda mol: mol.primary_score, reverse=reverse)
        n_random = int(self._n_oracle * self.epsilon.value)
        n_best = self._n_oracle - n_random
        oracle_mols = mols[:n_best]

        # In some rare cases we might receive fewer molecules than our acquisition size
        random_mols = (
            random.sample(mols[n_best:], k=n_random) if len(mols[n_best:]) >= n_random else []
        )
        oracle_mols.extend(random_mols)
        surrogate_mols = [mol for mol in mols[n_best:] if mol not in random_mols]
        self._send(oracle_mols=oracle_mols, surrogate_mols=surrogate_mols)


class ScheduledEpsilonGreedy(AcquisitionFunction):
    """Eps-Greedy acquisition function with scheduled decay"""

    inp_idx: Input[int] = Input(optional=True)
    """Current epoch input"""

    decay: Parameter[float] = Parameter(default=0.1)
    r"""
    Decay parameter describing how fast epsilon is reduced.
    Uses :math:`\epsilon_i = \epsilon_0 \frac{1}{1 + \beta i}`

    """

    epsilon0: Parameter[float] = Parameter(default=0.5)
    """Initial value for epsilon, i.e. the proportion of randomness"""

    def run(self) -> None:
        mols = self._recv()
        reverse = False
        if mols and (tag := mols[0].primary_score_tag):
            reverse = mols[0].score_agg[tag] == "max"
        mols.sort(key=lambda mol: mol.primary_score, reverse=reverse)

        epoch = self.inp_idx.receive() if self.inp_idx.ready() else 0
        epsilon = self.epsilon0.value * 1 / (1 + self.decay.value * epoch)
        n_random = int(self._n_oracle * epsilon)
        n_best = self._n_oracle - n_random
        oracle_mols = mols[:n_best]

        # In some rare cases we might receive fewer molecules than our acquisition size
        random_mols = random.sample(mols[n_best:], k=n_random) if mols[n_best:] else []
        oracle_mols.extend(random_mols)
        surrogate_mols = [mol for mol in mols[n_best:] if mol not in random_mols]
        self._send(oracle_mols=oracle_mols, surrogate_mols=surrogate_mols)


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper confidence bound (UCB) acquisition. Molecules must
    contain uncertainties under the `uncertainty` tag.

    """

    beta: Parameter[float] = Parameter(default=1.0)
    """Uncertainty balance"""

    def run(self) -> None:
        mols = self._recv()
        ucbs = {}
        for mol in mols:
            if mol.n_isomers == 0:
                error = 0.0
            else:
                error = max(float(cast(float, iso.get_tag("uncertainty"))) for iso in mol.molecules)
            ucbs[mol] = mol.primary_score + self.beta.value * error
        mols.sort(key=lambda mol: ucbs[mol])
        oracle_mols = mols[: self._n_oracle]
        surrogate_mols = mols[self._n_oracle :]
        self._send(oracle_mols=oracle_mols, surrogate_mols=surrogate_mols)


# TODO Expand to save multiple tags and use the correct aggregation for all of them
class FileCachedMol(Node):
    """Caches molecule scores in a dedicated file"""

    SCORE_TAG = "score"

    inp: Input[list[IsomerCollection]] = Input()
    """Unscored molecule input"""

    inp_calc: Input[list[IsomerCollection]] = Input()
    """Molecule input from a calculation node"""

    out: Output[list[IsomerCollection]] = Output()
    """Scored molecule output"""

    out_calc: Output[list[IsomerCollection]] = Output()
    """Output for molecules to be scored"""

    cache: FileParameter[Annotated[Path, Suffix("pkl")]] = FileParameter(exist_required=False)
    """Molecule scoring cache"""

    def run(self) -> None:
        cache: dict[str, tuple[str, float, Literal["min", "max"]]]

        scored = []
        to_score = []
        mols = self.inp.receive()

        if self.cache.filepath.exists():
            with self.cache.filepath.open("rb") as file:
                cache = pickle.load(file)
            self.logger.info("Loading existing cache with %s entries", len(cache))
        else:
            self.logger.info("Creating new cache")
            cache = {}

        # We lose the sorting here, but this shouldn't matter
        # as we have tagged each mol with its original index
        for mol in mols:
            if mol.smiles in cache:
                name, score, agg = cache[mol.smiles]
                mol.add_score(name, score, agg=agg)
                self.logger.info("Found cached score for '%s'", mol.smiles)
                scored.append(mol)
            else:
                self.logger.debug("Sending '%s' to be scored", mol.smiles)
                to_score.append(mol)

        if to_score:
            self.out_calc.send(to_score)
            newly_scored = self.inp_calc.receive()

            # Add new scores to the internal cache
            for mol in newly_scored:
                if mol.smiles is not None and mol.primary_score_tag is not None:
                    cache[mol.smiles] = (
                        mol.primary_score_tag,
                        mol.primary_score,
                        mol.score_agg[mol.primary_score_tag],
                    )
            scored.extend(newly_scored)
            self.logger.info("Updating cache with %s entries", len(newly_scored))

        self.out.send(scored)

        with self.cache.filepath.open("wb+") as file:
            pickle.dump(cache, file)


@deprecated("Please use single-iteration AL nodes instead")
class CachedMol(LoopedNode):
    """Caches molecule scores in memory"""

    SCORE_TAG = "score"

    inp: Input[list[IsomerCollection]] = Input()
    """Unscored molecule input"""

    inp_calc: Input[list[IsomerCollection]] = Input()
    """Molecule input from a calculation node"""

    out: Output[list[IsomerCollection]] = Output()
    """Scored molecule output"""

    out_calc: Output[list[IsomerCollection]] = Output()
    """Output for molecules to be scored"""

    _cache: dict[str, tuple[str, float, Literal["min", "max"]]]

    def prepare(self) -> None:
        self._cache = {}

    def run(self) -> None:
        scored = []
        to_score = []
        mols = self.inp.receive()
        self.logger.debug("Cache size: %s", len(self._cache))

        # We lose the sorting here, but this shouldn't matter
        # as we have tagged each mol with its original index
        for mol in mols:
            if mol.smiles in self._cache:
                name, score, agg = self._cache[mol.smiles]
                mol.add_score(name, score, agg=agg)
                self.logger.info("Found cached score for '%s'", mol.smiles)
                scored.append(mol)
            else:
                self.logger.debug("Sending '%s' to be scored", mol.smiles)
                to_score.append(mol)

        self.out_calc.send(to_score)

        # Add new scores to the internal cache
        newly_scored = self.inp_calc.receive()
        for mol in newly_scored:
            if mol.smiles is not None and mol.primary_score_tag is not None:
                self._cache[mol.smiles] = (
                    mol.primary_score_tag,
                    mol.primary_score,
                    mol.score_agg[mol.primary_score_tag],
                )

        scored.extend(newly_scored)
        self.out.send(scored)


@deprecated("Please use single-iteration AL nodes instead")
class ActiveLearning(Graph):
    """Active learning for molecules"""

    inp: Input[list[IsomerCollection]]
    """Main molecule input"""

    inp_extra: Input[list[IsomerCollection]]
    """Additional molecules to assist training proxy"""

    out_oracle: Output[list[IsomerCollection]]
    """Output for the oracle"""

    inp_oracle: Input[list[IsomerCollection]]
    """Input from the oracle"""

    out_acq: Output[list[IsomerCollection]]
    """Output for the acquisition function"""

    inp_acq_oracle: Input[list[IsomerCollection]]
    """Input from the acquisition function for oracle calls"""

    inp_acq_surrogate: Input[list[IsomerCollection]]
    """Input from the acquisition function for surrogate model calls"""

    out: Output[list[IsomerCollection]]
    """Scored molecule output"""

    n_train: Parameter[int]
    """Number of molecules to use for retraining"""

    epochs: Parameter[list[int]]
    """Number of warmup, pooling, and production iterations"""

    hyper_freq: Parameter[int]
    """Frequency of performing a new hyperparameter optimization"""

    proxy_model: FileParameter[Path]
    """Location of the built Qptuna model file"""

    proxy_config: FileParameter[Path]
    """Qptuna configuration for hyperparameter search"""

    proxy_pool: FileParameter[Annotated[Path, Suffix("csv")]]
    """Location of the training pool"""

    def build(self) -> None:
        epoch = self.add(TimeDistribute[list[IsomerCollection]], name="epoch")
        multi = self.add(Multiplex[list[IsomerCollection]])
        copy_pool = self.add(Copy[list[IsomerCollection]], name="copy-pool")
        merge = self.add(Merge[list[IsomerCollection]], name="merge-all")
        merge_train = self.add(Merge[list[IsomerCollection]], name="merge-train")
        iso_merge = self.add(MergeLists[IsomerCollection], name="merge-isomers")

        self.connect_all(
            # Warmup
            (epoch.out, multi.inp),
            (multi.out, merge.inp),
            # Pooling
            (epoch.out, multi.inp),
            (multi.out, copy_pool.inp),
            (copy_pool.out, merge.inp),
            (copy_pool.out, merge_train.inp),
        )

        barrier = self.add(Barrier[list[IsomerCollection]])
        tuna_predict = self.add(QptunaPredict, loop=True)
        copy_al = self.add(Copy[list[IsomerCollection]], name="copy-al")

        # AL, scoring
        self.connect_all(
            (epoch.out, barrier.inp),
            (barrier.out, tuna_predict.inp),
            (multi.out, copy_al.inp),
            (copy_al.out, iso_merge.inp),
        )

        hyper_dec = self.add(CopyEveryNIter[list[IsomerCollection]], loop=True)
        tuna_hyper = self.add(QptunaHyper, loop=True)
        tuna_train = self.add(QptunaTrain, loop=True)
        iso_sort = self.add(SortByTag, loop=True)
        iso_sort.tag.set("idx")

        # AL, training
        self.connect_all(
            (iso_merge.out, iso_sort.inp),
            (iso_sort.out, merge.inp),
            (copy_al.out, merge_train.inp),
            (hyper_dec.out, tuna_train.inp),
            (hyper_dec.out, tuna_hyper.inp),
            (tuna_hyper.out, tuna_train.inp_config),
        )

        # Extras
        cache = self.add(CachedMol, loop=True)
        inject = self.add(MergeLists[IsomerCollection], name="inject")
        self.connect_all(
            (tuna_train.out, barrier.inp_signal),
            (multi.out_single, cache.inp),
            (cache.out, multi.inp_single),
            (merge_train.out, inject.inp),
            (inject.out, hyper_dec.inp),
        )

        self.map(epoch.inp, merge.out)

        inject.inp.cached = True
        inject.inp.optional = True
        self.map_port(inject.inp, name="inp_extra")
        self.map_port(cache.out_calc, name="out_oracle")
        self.map_port(cache.inp_calc, name="inp_oracle")
        self.map_port(tuna_predict.out, name="out_acq")
        self.map_port(multi.inp, name="inp_acq_oracle")
        self.map_port(iso_merge.inp, name="inp_acq_surrogate")

        self.combine_parameters(
            tuna_train.model, tuna_predict.model, tuna_hyper.model, name="proxy_model"
        )
        self.combine_parameters(tuna_train.pool, tuna_hyper.pool, name="proxy_pool")
        self.combine_parameters(tuna_train.n_train, tuna_hyper.n_train)
        self.combine_parameters(tuna_hyper.configuration, name="proxy_config")
        self.combine_parameters(hyper_dec.freq, name="hyper_freq")
        self.combine_parameters(epoch.pattern, default=[0, 1, -1], name="epochs")


class ALPooling(Graph):
    """Pooling subgraph for active learning"""

    inp: Input[list[IsomerCollection]]
    """Scored molecule input"""

    out: Output[list[IsomerCollection]]
    """Scored molecule output"""

    inp_extra: Input[list[IsomerCollection]]
    """Additional molecules to assist training proxy"""

    out_config: Output[Annotated[Path, Suffix("json")]]
    """Build config output"""

    n_train: Parameter[int]
    """Number of molecules to use for retraining"""

    proxy_model: FileParameter[Path]
    """Location of the built Qptuna model file"""

    proxy_config: FileParameter[Path]
    """Qptuna configuration for hyperparameter search"""

    proxy_pool: FileParameter[Annotated[Path, Suffix("csv")]]
    """Location of the training pool"""

    def build(self) -> None:
        copy_pool = self.add(Copy[list[IsomerCollection]], name="copy-pool")
        copy_tuna = self.add(Copy[list[IsomerCollection]], name="copy-tuna")
        barrier = self.add(Barrier[list[IsomerCollection]])
        inject = self.add(MergeLists[IsomerCollection], name="inject")
        hyper = self.add(QptunaHyper)
        train = self.add(QptunaTrain)
        copy_config = self.add(Copy[Annotated[Path, Suffix("json")]], name="copy-config")

        # Data distribution
        self.connect_all(
            (copy_pool.out, barrier.inp),
            (copy_pool.out, inject.inp),
            (inject.out, copy_tuna.inp),
        )

        # (Hyperparameter) training
        self.connect_all(
            (copy_tuna.out, hyper.inp),
            (copy_tuna.out, train.inp),
            (hyper.out, copy_config.inp),
            (copy_config.out, train.inp_config),
            (train.out, barrier.inp_signal),
        )

        self.map(copy_pool.inp, barrier.out)

        # Unknown type problems here...
        self.inp_extra = self.map_port(inject.inp, name="inp_extra")  # type: ignore
        self.out_config = self.map_port(copy_config.out, name="out_config")  # type: ignore
        self.combine_parameters(train.n_train, hyper.n_train)
        self.combine_parameters(train.model, hyper.model, name="proxy_model")
        self.combine_parameters(hyper.configuration, name="proxy_config")
        self.combine_parameters(train.pool, hyper.pool, name="proxy_pool")


class AL(Graph):
    """Production active learning"""

    inp_proxy: Input[list[IsomerCollection]]
    """Predicted molecule input"""

    inp_oracle: Input[list[IsomerCollection]]
    """Calculated molecule input"""

    out: Output[list[IsomerCollection]]
    """Scored molecule output"""

    inp_extra: Input[list[IsomerCollection]]
    """Additional molecules to assist training proxy"""

    inp_config: Input[Annotated[Path, Suffix("json")]]
    """Build config input"""

    n_train: Parameter[int]
    """Number of molecules to use for retraining"""

    proxy_model: FileParameter[Path]
    """Location of the built Qptuna model file"""

    proxy_pool: FileParameter[Annotated[Path, Suffix("csv")]]
    """Location of the training pool"""

    def build(self) -> None:
        copy_oracle = self.add(Copy[list[IsomerCollection]], name="copy-oracle")
        merge_all = self.add(MergeLists[IsomerCollection], name="merge-all")
        iso_sort = self.add(SortByTag)
        train = self.add(QptunaTrain)
        barrier = self.add(Barrier[list[IsomerCollection]])

        self.connect_all(
            (copy_oracle.out, merge_all.inp),
            (copy_oracle.out, train.inp),
            (train.out, barrier.inp_signal),
            (merge_all.out, iso_sort.inp),
            (iso_sort.out, barrier.inp),
        )

        self.map_port(copy_oracle.inp, name="inp_oracle")
        self.map_port(merge_all.inp, name="inp_proxy")
        self.map_port(barrier.out, name="out")
        self.map_port(train.inp_config, name="inp_config")

        self.map(train.n_train)
        self.combine_parameters(train.model, name="proxy_model")
        self.combine_parameters(train.pool, name="proxy_pool")
        iso_sort.tag.set("idx")


class ActiveLearningSingle(Graph):
    """Active learning for molecules (non-looped)"""

    inp: Input[list[IsomerCollection]]
    """Main molecule input"""

    inp_iter: Input[int] = Input()
    """Current AL iteration"""

    inp_extra: Input[list[IsomerCollection]]
    """Additional molecules to assist training proxy"""

    out_oracle: Output[list[IsomerCollection]]
    """Output for the oracle"""

    inp_oracle: Input[list[IsomerCollection]]
    """Input from the oracle"""

    out_acq: Output[list[IsomerCollection]]
    """Output for the acquisition function"""

    inp_acq_oracle: Input[list[IsomerCollection]]
    """Input from the acquisition function for oracle calls"""

    inp_acq_surrogate: Input[list[IsomerCollection]]
    """Input from the acquisition function for surrogate model calls"""

    out: Output[list[IsomerCollection]]
    """Scored molecule output"""

    active: Flag
    """Whether active learning is active"""

    n_train: Parameter[int]
    """Number of molecules to use for retraining"""

    epochs: Parameter[list[int]]
    """Number of warmup, pooling, and production iterations"""

    hyper_freq: Parameter[int]
    """Frequency of performing a new hyperparameter optimization"""

    cache: FileParameter[Path]
    """Location of the molecule score cache"""

    proxy_model: FileParameter[Path]
    """Location of the built Qptuna model file"""

    proxy_config: FileParameter[Path]
    """Qptuna configuration for hyperparameter search"""

    proxy_build: FileParameter[Path]
    """Qptuna build configuration"""

    proxy_pool: FileParameter[Annotated[Path, Suffix("csv")]]
    """Location of the training pool"""

    aggregation: Parameter[Literal["min", "max"]]
    """
    What kind of aggregation to use for the score, i.e. ``min``
    if lower scores are better, ``max`` if higher scores are better

    """

    def build(self) -> None:
        index = self.add(IntegerMap, name="map")
        choice = self.add(Choice[int], name="al-choice")
        false = self.add(LoadData[bool], name="false", parameters={"data": False})
        switch = self.add(LoadData[bool], name="switch", parameters={"data": True})
        tagidx = self.add(TagIndex)
        epoch = self.add(IndexDistribute[list[IsomerCollection]], name="epoch")
        multi = self.add(Multiplex[list[IsomerCollection]])
        merge = self.add(Merge[list[IsomerCollection]], name="merge-all")
        tuna_predict = self.add(QptunaPredict)
        config_buffer = self.add(FileBuffer[Annotated[Path, Suffix("json")]])
        pool = self.add(ALPooling)
        prod = self.add(AL)

        self.connect_all(
            (epoch.out, multi.inp),
            (multi.out, merge.inp),
            (epoch.out, multi.inp),
            (multi.out, pool.inp),
            (epoch.out, tuna_predict.inp),
            (multi.out, prod.inp_oracle),
        )

        # For maize, bool and int are compatible, but mypy disagrees
        self.connect_all(  # type: ignore
            (switch.out, choice.inp_index),
            (false.out, choice.inp),
        )

        self.connect_all(
            (tagidx.out, epoch.inp),
            (index.out, choice.inp),
            (choice.out, epoch.inp_index),
            (pool.out, merge.inp),
            (prod.out, merge.inp),
        )

        # Extras
        cache = self.add(FileCachedMol)
        self.connect_all(
            (pool.out_config, config_buffer.inp),
            (config_buffer.out, prod.inp_config),
            (multi.out_single, cache.inp),
            (cache.out, multi.inp_single),
        )

        self.map(tagidx.inp, merge.out)
        self.combine_parameters(tuna_predict.agg, name="aggregation")

        self.map_port(index.inp, name="inp_iter")
        self.map_port(pool.inp_extra, name="inp_extra")
        self.map_port(cache.out_calc, name="out_oracle")
        self.map_port(cache.inp_calc, name="inp_oracle")
        self.map_port(tuna_predict.out, name="out_acq")
        self.map_port(multi.inp, name="inp_acq_oracle")
        self.map_port(prod.inp_proxy, name="inp_acq_surrogate")

        self.active = self.combine_parameters(switch.data, name="active", default=True)
        self.n_train = self.combine_parameters(pool.n_train, prod.n_train)
        self.combine_parameters(cache.cache, name="cache")
        self.combine_parameters(
            pool.proxy_model, tuna_predict.model, prod.proxy_model, name="proxy_model"
        )
        self.combine_parameters(pool.proxy_pool, prod.proxy_pool, name="proxy_pool")
        self.combine_parameters(pool.proxy_config, name="proxy_config")
        self.combine_parameters(config_buffer.file, name="proxy_build")
        self.combine_parameters(index.pattern, default=[0, 1, -1], name="epochs")


@pytest.fixture
def mols() -> list[IsomerCollection]:
    smiles = [
        "Nc1ncnc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cc(OC)c(OC)c(c3)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC",
        "Nc1nc(F)nc(c12)n(CCCC#C)c(n2)Cc3cc(OC)c(OC)c(c3Cl)OC",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3",
        "Nc1ncnc(c12)n(CCCC)c(n2)Cc(cc3)cc(c34)OCO4",
        "Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc3cc(OC)ccc3OC",
        "Nc1nc(F)nc(c12)n(CCCC)c(n2)Cc(cc3)cc(c34)OCO4",
        "Nc1nc(F)nc(c12)n(CCCC#C)c(n2)Cc3cc(OC)ccc3OC",
    ]

    return [IsomerCollection.from_smiles(smi) for smi in smiles]


@pytest.fixture
def mols_with_scores(mols: list[IsomerCollection]) -> list[IsomerCollection]:
    for mol in mols:
        for isomer in mol.molecules:
            isomer.add_score("score", -10 * np.random.random(10))
    return mols


@pytest.fixture
def mols_with_uncertainty(mols_with_scores: list[IsomerCollection]) -> list[IsomerCollection]:
    for mol in mols_with_scores:
        for iso in mol.molecules:
            iso.set_tag("uncertainty", np.random.normal())
    return mols_with_scores


class TestSuiteAcquisition:
    def test_Random(self, mols_with_scores: list[IsomerCollection]) -> None:
        rig = TestRig(Random)
        res = rig.setup_run(inputs={"inp": [mols_with_scores]}, parameters={"n_oracle": 5})
        oracle_mols = res["out_oracle"].get()
        assert oracle_mols is not None
        assert len(oracle_mols) == 5

        surrogate_mols = res["out_surrogate"].get()
        assert surrogate_mols is not None
        assert len(surrogate_mols) == len(mols_with_scores) - 5

    def test_Random_prop(self, mols_with_scores: list[IsomerCollection]) -> None:
        rig = TestRig(Random)
        res = rig.setup_run(inputs={"inp": [mols_with_scores]}, parameters={"n_oracle": 0.3})
        oracle_mols = res["out_oracle"].get()
        assert oracle_mols is not None
        assert len(oracle_mols) == int(len(mols_with_scores) * 0.3)

        surrogate_mols = res["out_surrogate"].get()
        assert surrogate_mols is not None
        assert len(surrogate_mols) == int(len(mols_with_scores) * 0.7)

    def test_Greedy(self, mols_with_scores: list[IsomerCollection]) -> None:
        rig = TestRig(Greedy)
        res = rig.setup_run(inputs={"inp": [mols_with_scores]}, parameters={"n_oracle": 5})
        oracle_mols = res["out_oracle"].get()
        assert oracle_mols is not None
        assert len(oracle_mols) == 5
        surrogate_mols = res["out_surrogate"].get()
        assert surrogate_mols is not None
        assert len(surrogate_mols) == len(mols_with_scores) - 5
        assert np.mean([mol.primary_score for mol in oracle_mols]) < np.mean(
            [mol.primary_score for mol in surrogate_mols]
        )

    def test_EpsilonGreedy(self, mols_with_scores: list[IsomerCollection]) -> None:
        rig = TestRig(EpsilonGreedy)
        res = rig.setup_run(inputs={"inp": [mols_with_scores]}, parameters={"n_oracle": 5})
        oracle_mols = res["out_oracle"].get()
        assert oracle_mols is not None
        assert len(oracle_mols) == 5
        surrogate_mols = res["out_surrogate"].get()
        assert surrogate_mols is not None
        assert len(surrogate_mols) == len(mols_with_scores) - 5
        assert np.mean([mol.primary_score for mol in oracle_mols]) < np.mean(
            [mol.primary_score for mol in surrogate_mols]
        )

    def test_ScheduledEpsilonGreedy(self, mols_with_scores: list[IsomerCollection]) -> None:
        rig = TestRig(ScheduledEpsilonGreedy)
        res = rig.setup_run(
            inputs={"inp": [mols_with_scores], "inp_idx": 5}, parameters={"n_oracle": 5}
        )
        oracle_mols = res["out_oracle"].get()
        assert oracle_mols is not None
        assert len(oracle_mols) == 5
        surrogate_mols = res["out_surrogate"].get()
        assert surrogate_mols is not None
        assert len(surrogate_mols) == len(mols_with_scores) - 5
        assert np.mean([mol.primary_score for mol in oracle_mols]) < np.mean(
            [mol.primary_score for mol in surrogate_mols]
        )

    def test_UpperConfidenceBound(self, mols_with_uncertainty: list[IsomerCollection]) -> None:
        rig = TestRig(UpperConfidenceBound)
        res = rig.setup_run(inputs={"inp": [mols_with_uncertainty]}, parameters={"n_oracle": 5})
        oracle_mols = res["out_oracle"].get()
        assert oracle_mols is not None
        assert len(oracle_mols) == 5
        surrogate_mols = res["out_surrogate"].get()
        assert surrogate_mols is not None
        assert len(surrogate_mols) == len(mols_with_uncertainty) - 5
        assert np.mean([mol.primary_score for mol in oracle_mols]) < np.mean(
            [mol.primary_score for mol in surrogate_mols]
        )


class TestSuiteAL:
    def test_FileCachedMol(
        self,
        mols: list[IsomerCollection],
        mols_with_scores: list[IsomerCollection],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache.pkl"
        rig = TestRig(FileCachedMol)
        res = rig.setup_run(
            inputs={"inp": [mols], "inp_calc": [mols_with_scores]}, parameters={"cache": cache}
        )
        mols1 = res["out_calc"].get()
        assert mols1 is not None
        assert len(mols1) == len(mols)

        mols_out = res["out"].get()
        assert mols_out is not None
        assert len(mols_out) == len(mols_with_scores)
        assert all(mol.primary_score for mol in mols_out)

        rig = TestRig(FileCachedMol)
        res = rig.setup_run(inputs={"inp": [mols], "inp_calc": []}, parameters={"cache": cache})
        mols_out = res["out"].get()
        assert mols_out is not None
        assert len(mols_out) == len(mols_with_scores)
        assert all(mol.primary_score for mol in mols_out)

    def test_CachedMol(
        self, mols: list[IsomerCollection], mols_with_scores: list[IsomerCollection]
    ) -> None:
        rig = TestRig(CachedMol)
        res = rig.setup_run(
            inputs={"inp": [mols, mols], "inp_calc": [mols_with_scores, []]}, max_loops=2
        )
        mols1 = res["out_calc"].get()
        assert mols1 is not None
        assert len(mols1) == len(mols)

        mols_out = res["out"].get()
        assert mols_out is not None
        assert len(mols_out) == len(mols_with_scores)
        assert all(mol.primary_score for mol in mols_out)

        mols2 = res["out_calc"].get()
        assert mols2 is not None
        assert len(mols2) == 0

        mols_out = res["out"].get()
        assert mols_out is not None
        assert len(mols_out) == len(mols_with_scores)
        assert all(mol.primary_score for mol in mols_out)

    def test_ActiveLearningProgress(
        self, mols_with_scores: list[IsomerCollection], temp_working_dir: Path
    ) -> None:
        rig = TestRig(ActiveLearningProgress)
        res = rig.setup_run(
            inputs={"inp": [mols_with_scores]},
            parameters={"save_epochs": True, "save_location": Path()},
        )
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == len(mols_with_scores)
        assert (Path() / "epoch-0.csv").exists()
