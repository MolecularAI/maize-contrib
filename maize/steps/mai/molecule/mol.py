"""Molecule handling steps"""

# pylint: disable=import-outside-toplevel, import-error

from collections import defaultdict
from copy import deepcopy
import csv
import itertools
import json
from pathlib import Path
import random
import re
import shutil
from typing import Annotated, Any, Callable, Iterable, List, Literal, TypeVar

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, FileParameter, Suffix, Flag, MultiInput
from maize.utilities.chem.chem import ChemistryException, ValidRDKitTagType
from maize.utilities.chem import (
    IsomerCollection,
    Isomer,
    load_sdf_library,
    save_sdf_library,
    merge_isomers,
)
from maize.utilities.testing import TestRig
from maize.utilities.io import Config
from maize.steps.mai.molecule.compchem_utils import (
    Loader
)
from maize.steps.mai.molecule.template_repo import get_chg_and_mult, ReactionComponent
from rdkit import Chem
NumericType = int | float | np.number[Any]
T = TypeVar("T")


class Smiles2Molecules(Node):
    """
    Converts SMILES codes into a set of molecules with distinct
    isomers and conformers using the RDKit embedding functionality.

    See Also
    --------
    :class:`~maize.steps.mai.molecule.Gypsum` :
        A more advanced procedure for producing different
        isomers and high-energy conformers.

    """
    tags = {"chemistry", "sampler", "embedding"}

    inp: Input[list[str]] = Input()
    """SMILES input"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output"""

    n_conformers: Parameter[int] = Parameter(default=1)
    """Number of conformers to generate"""

    n_variants: Parameter[int] = Parameter(default=1)
    """Maximum number of stereoisomers to generate"""

    embed: Flag = Flag(default=True)
    """
    Whether to create embeddings for the molecule. May not be
    required if passing it on to another embedding system.

    """

    def run(self) -> None:
        smiles = self.inp.receive()
        mols: list[IsomerCollection] = []
        n_variants = self.n_variants.value if self.embed.value else 0
        for i, smi in enumerate(smiles):
            self.logger.info("Embedding %s/%s ('%s')", i + 1, len(smiles), smi.strip())
            try:
                mol = IsomerCollection.from_smiles(smi, max_isomers=n_variants)
                if self.embed.value:
                    mol.embed(self.n_conformers.value)
            except ChemistryException as err:
                self.logger.warning("Unable to create '%s' (%s), not sanitizing...", smi, err)
                if "SMILES Parse Error" in err.args[0]:
                    mol = IsomerCollection([])
                    mol.smiles = smi
                else:
                    mol = IsomerCollection.from_smiles(smi, max_isomers=0, sanitize=False)
            mols.append(mol)
        self.out.send(mols)


class Mol2Isomers(Node):
    """Convert a list of `IsomerCollection` to a list of `Isomer`"""

    tags = {"chemistry", "utility", "conversion"}

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule input"""

    out: Output[list[Isomer]] = Output()
    """Isomer output"""

    def run(self) -> None:
        mols = self.inp.receive()
        self.out.send(list(itertools.chain(*[mol.molecules for mol in mols])))


class Isomers2Mol(Node):
    """Convert a list of `Isomer` to a list of `IsomerCollection`"""

    tags = {"chemistry", "utility", "conversion"}

    inp: Input[list[Isomer]] = Input()
    """Molecule input"""

    out: Output[list[IsomerCollection]] = Output()
    """Isomer output"""

    combine: Parameter[bool] = Parameter(default=False)
    """Should all the isomers be combined into a single `IsomerCollection`"""

    def run(self) -> None:
        mols = self.inp.receive()
        if self.combine.value:
            iso_collection = IsomerCollection(mols)
            self.out.send([iso_collection])
        else:
            self.out.send([IsomerCollection([mol]) for mol in mols])


class SaveMolecule(Node):
    """Save a molecule to an SDF file."""

    tags = {"chemistry", "utility", "saving"}

    inp: Input[IsomerCollection] = Input()
    """Molecule input"""

    path: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(exist_required=False)
    """SDF output destination"""

    def run(self) -> None:
        mol = self.inp.receive()
        self.logger.info("Received '%s'", mol)
        mol.to_sdf(self.path.value)


class File2Molecule(Node):
    """Load a molecule from an SDF, MAE or MAEGZ file."""

    tags = {"chemistry", "utility", "saving", "conversion"}

    inp: Input[Annotated[Path, Suffix("sdf", "mae", "maegz")]] = Input(mode="copy")
    """Path to the SDF, MAE or MAEGZ file"""

    out: Output[Isomer] = Output()
    """Isomer output"""

    def run(self) -> None:
        input_path = self.inp.receive()
        print(input_path)
        if input_path.suffix == ".mae":
            mol = IsomerCollection.from_mae(input_path)
        elif input_path.suffix == ".maegz":
            mol = IsomerCollection.from_maegz(input_path)
        elif input_path.suffix == ".sdf":
            mol = IsomerCollection.from_sdf(input_path)
        else:
            raise ValueError("incorrect filetype %s" % str(input_path.suffix))
        self.out.send(mol.molecules[0])


class LoadMolecule(Node):
    """Load a molecule from an SDF or MAE file."""

    tags = {"chemistry", "utility", "loading"}

    out: Output[Isomer] = Output()
    """Isomer output"""

    path: FileParameter[Annotated[Path, Suffix("sdf", "mae", "maegz")]] = FileParameter()
    """Path to the SDF or MAE file"""

    tag: Parameter[str] = Parameter(optional=True)
    """If set, will use this tag as the primary score for the isomer"""

    agg: Parameter[Literal["min", "max"]] = Parameter(default="min")
    """How to aggregate scores for the score given by ``tag``"""

    def run(self) -> None:
        input_path = self.path.filepath
        if input_path.suffix == ".mae":
            mol = IsomerCollection.from_mae(input_path)
        elif input_path.suffix == ".maegz":
            mol = IsomerCollection.from_maegz(input_path)
        elif input_path.suffix == ".sdf":
            mol = IsomerCollection.from_sdf(input_path)
        else:
            raise ValueError("incorrect filetype %s" % str(input_path.suffix))
        if self.tag.is_set:
            mol.molecules[0].add_score_tag(self.tag.value, agg=self.agg.value)
        self.out.send(mol.molecules[0])


class LoadSingleRow(Node):
    """Load a single row from a CSV file"""

    tags = {"chemistry", "utility", "loading"}

    out: Output[pd.core.series.Series] = Output()
    """String output"""

    path: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter()
    """Path to the CSV file"""

    read_csv_parameters: Parameter[dict[str, Any]] = Parameter(default_factory=dict)
    """This holds all the keyworded parameters that you might want to use for your read_csv calls"""

    index: Parameter[int] = Parameter()
    """Index of row to read"""

    def run(self) -> None:
        csv_file = pd.read_csv(
            filepath_or_buffer=self.path.filepath, **self.read_csv_parameters.value
        )
        row = csv_file.iloc[self.index.value]
        self.out.send(row)

class LoadSmilesAsIsomerCollection(Node):
    """Load a single smiles from a CSV file and convert it to an isomercollection"""

    out: Output[list[IsomerCollection]] = Output()

    path: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter()
    """Path to the CSV file"""

    read_csv_parameters: Parameter[dict[str, Any]] = Parameter(default_factory=dict)
    """This holds all the keyworded parameters that you might want to use for your read_csv calls"""

    index: Parameter[int] = Parameter()
    """Index of row to read"""

    smiles_field_name: Parameter[str] = Parameter(default="")
    """Name of the field in the row to read the smiles from"""

    def run(self) -> None:
        csv_file = pd.read_csv(
            filepath_or_buffer=self.path.filepath, **self.read_csv_parameters.value
        )
        row = csv_file.iloc[self.index.value]
        smiles = row[self.smiles_field_name.value]
        self.logger.info(f"Loaded smiles: {smiles}")

        smiles_rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        smiles_mol_obj = Loader.molecule_from_rdkit(smiles_rdmol)
        mol_name = "main_molecule"
        smiles_mol_obj.name = smiles
        component = ReactionComponent(component=mol_name,
                                        smiles=smiles,
                                        parameters = get_chg_and_mult(smiles_rdmol))
        component.structure.append(smiles_mol_obj)
        iso_collection_list = [IsomerCollection([isomer for isomer in component.to_isomer()])]
        
        self.logger.info("Sending smiles back from LoadSmilesAsIsomerCollection")
        self.out.send(iso_collection_list)

class IsomerCollectionSaving(Node):
    """Saves isomer collections in a specific directory structure"""

    tags = {"chemistry", "utility", "saving"}

    inp: Input[list[IsomerCollection]] = Input()
    """list of IsomerCollection input"""

    isomer_output_location: FileParameter[Path] = FileParameter(exist_required=False)
    """location where to save the JSON files of the isomers"""

    def run(self) -> None:
        isomers = self.inp.receive()

        isomer_location = Path(self.isomer_output_location.value)
        self.logger.info(isomer_location)
        if isomer_location.exists():
            shutil.rmtree(isomer_location)
        isomer_location.mkdir()

        for isomercollection_index, isomercollection in enumerate(isomers):
            isomercollection_path = isomer_location / f"mol-{isomercollection_index}"
            if isomercollection_path.exists():
                shutil.rmtree(isomercollection_path)
            isomercollection_path.mkdir()

            for isomer_index, isomer in enumerate(isomercollection.molecules):
                isomer_path = isomercollection_path / f"iso-{isomer_index}"

                if isomer_path.exists():
                    shutil.rmtree(isomer_path)
                isomer_path.mkdir()

                result_file = isomer_path / f"{isomer.name}_results"
                with open(result_file, "w") as file:
                    json.dump(isomer.tags, file)


class LoadSmiles(Node):
    """Load SMILES codes from a ``.smi`` file."""

    tags = {"chemistry", "utility", "loading"}

    path: FileParameter[Annotated[Path, Suffix("smi")]] = FileParameter()
    """SMILES file input"""

    out: Output[list[str]] = Output()
    """SMILES output"""

    sample: Parameter[int] = Parameter(optional=True)
    """Take a sample of SMILES"""

    def run(self) -> None:
        with self.path.filepath.open() as file:
            smiles = [smi.strip("\n") for smi in file.readlines()]
            if self.sample.is_set:
                smiles = random.choices(smiles, k=self.sample.value)
            self.out.send(smiles)


class ExtractTag(Node):
    """Pull a tag from an Isomer"""

    tags = {"chemistry", "utility"}

    inp: Input[Isomer] = Input()
    """A isomer to extract tag from"""

    out: Output[ValidRDKitTagType] = Output()
    """value of the tag"""

    tag_to_extract: Parameter[str] = Parameter(optional=True)
    """tag to export, will use score_tag by default"""

    def run(self) -> None:
        isom = self.inp.receive()

        if self.tag_to_extract.is_set:
            extract_tag = self.tag_to_extract.value
            if isom.has_tag(extract_tag):
                self.out.send(isom.get_tag(extract_tag))
            else:
                self.logger.debug("provided isomer does not have tag %s" % extract_tag)
                self.out.send(np.nan)

        else:
            self.out.send(isom.primary_score)


class ToSmiles(Node):
    """transform an isomer or IsomerCollection (or list thereof) to a list of SMILES"""

    tags = {"chemistry", "utility", "conversion"}

    inp: Input[Isomer | IsomerCollection | List[IsomerCollection] | List[Isomer]] = Input()
    """SMILES output"""

    out: Output[List[str]] = Output()
    """SMILES output"""

    def run(self) -> None:
        def _liststrip(maybe_list: list[T] | T) -> T:
            # need this uglyness to handle
            # both list of isomer and list of isomer comp
            if isinstance(maybe_list, list):
                return maybe_list[0]
            else:
                return maybe_list

        input_data = self.inp.receive()
        smiles: list[str] | str
        if isinstance(input_data, list):  # need to iteratively build
            smiles = [_liststrip(iso.to_smiles()) for iso in input_data]
        else:
            smiles = input_data.to_smiles()
        if isinstance(smiles, str):  # catch the case where used with single isomer
            smiles = [smiles]
        self.logger.info("sending %i smiles: %s" % (len(smiles), " ".join(smiles)))

        self.out.send(smiles)


class SaveSingleLibrary(Node):
    """Save a list of molecules to a single SDF file."""

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule library input"""

    file: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(exist_required=False)
    """Save location"""

    output_tags: Parameter[list[str]] = Parameter(optional=True)
    """Tags to write out, will write out all tags by default"""

    all_conformers: Flag = Flag(default=False)
    """If ``True`` will write all conformers instead of only the first one"""

    increment: Flag = Flag(default=True)
    """Whether to add an index to each filename if it already exists"""

    def run(self) -> None:
        mols = self.inp.receive()
        tags = self.output_tags.value if self.output_tags.is_set else None

        # Increment filename if it already exists, useful for REINVENT
        base = file = self.file.filepath
        if self.increment.value:
            i = 1
            while file.exists():
                file = base.parent / f"{base.stem}-{i}{base.suffix}"
                i += 1

        save_sdf_library(
            file,
            mols,
            conformers=self.all_conformers.value,
            tags=tags,
            split_strategy="schrodinger",
        )


class SaveLibrary(Node):
    """Save a list of molecules to multiple SDF files."""

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule library input"""

    base_path: FileParameter[Path] = FileParameter(exist_required=False)
    """Base output file path name without a suffix, i.e. /path/to/output"""

    output_tags: Parameter[list[str]] = Parameter(optional=True)
    """Tags to write out"""

    def run(self) -> None:
        mols = self.inp.receive()
        base = self.base_path.value

        for i, mol in enumerate(mols):
            file = base.with_name(f"{base.name}{i}.sdf")
            if self.output_tags.is_set:
                # this if statement is better than using mol[0].tags
                # as a default since it would support ragged tags
                mol.to_sdf(file, tags=self.output_tags.value)
            else:
                mol.to_sdf(file)


class SaveIsomers(Node):
    """Save a list of molecules to a single SDF file."""

    tags = {"chemistry", "utility", "saving"}

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule library input"""

    file: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(exist_required=False)
    """Location of the SDF library file"""

    append: Flag = Flag(default=False)
    """Whether to append to the file instead of overwriting"""

    def run(self) -> None:
        mols = self.inp.receive()
        save_sdf_library(self.file.filepath, mols, conformers=False, append=self.append.value)
        self.logger.info("Saved %s molecules to %s", len(mols), self.file.filepath)


class SaveScores(Node):
    """Save VINA Scores to a JSON file."""

    tags = {"chemistry", "utility", "saving"}

    inp: Input[NDArray[np.float32]] = Input()
    """Molecule input"""

    path: FileParameter[Annotated[Path, Suffix("json")]] = FileParameter(exist_required=False)
    """JSON output destination"""

    def run(self) -> None:
        scores = self.inp.receive()
        self.logger.info(f"Received #{len(scores):d} scores")
        with open(self.path.value, "w") as f:
            json.dump(list(scores), f)


class LoadLibrary(Node):
    """Load a small molecule library from an SDF file"""

    tags = {"chemistry", "utility", "loading"}

    path: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter()
    """Input SDF file"""

    out: Output[list[IsomerCollection]] = Output()
    """Molecule output, each entry in the SDF is parsed as a separate molecule"""

    score_tag: Parameter[str] = Parameter(optional=True)
    """SDF tag used to set scores for the loaded library"""

    score_agg: Parameter[Literal["min", "max"]] = Parameter(default="min")
    """The type of aggregation to use for the score"""

    renumber: Flag = Flag(default=False)
    """Whether to renumber the atoms"""

    def run(self) -> None:
        mols = load_sdf_library(
            self.path.filepath, split_strategy="none", renumber=self.renumber.value
        )
        if self.score_tag.is_set:
            for mol in mols:
                for iso in mol.molecules:
                    iso.add_score_tag(self.score_tag.value, agg=self.score_agg.value)
        self.out.send(mols)


def _get_first_iso(mols: list[IsomerCollection]) -> Isomer | None:
    """Safely gets the first available isomer"""
    if mols and mols[0].molecules:
        return mols[0].molecules[0]
    return None


class CombineMolecules(Node):
    """Combine molecules scored using different methods together"""

    tags = {"chemistry", "utility"}

    inp: MultiInput[list[IsomerCollection]] = MultiInput()
    """Multiple scored molecule inputs"""

    out: Output[list[IsomerCollection]] = Output()
    """Combined scores"""

    upstream_names: Parameter[list[str]] = Parameter(optional=True)
    """Names of the individual upstream branches"""

    def run(self) -> None:
        # Collect all libraries from all branches
        branches: list[list[IsomerCollection]] = []
        for i, inp in enumerate(self.inp):
            self.logger.debug("Attempting to receive batch %s", i)
            mols = inp.receive()
            if (iso := _get_first_iso(mols)) is not None:
                self.logger.info("Received %s from %s", mols, iso.get_tag("origin", "N/A"))
            branches.append(mols)

        # Determine names for each branch
        names = (
            [f"upstream-{i}" for i, _ in enumerate(self.inp)]
            if not self.upstream_names.is_set
            else self.upstream_names.value
        )

        # Build a flat library of all isomers first
        isos: dict[str, Isomer] = {}
        for name, branch in zip(names, branches):
            for mol in branch:
                for iso in mol.molecules:
                    iso.uniquify_tags(fallback=name)
                    if iso.inchi in isos:
                        iso = merge_isomers(isos[iso.inchi], iso)
                    isos[iso.inchi] = iso

        # Group by molecule / IsomerCollection / topology
        master: dict[str, list[Isomer]] = defaultdict(list)
        for inchi, iso in isos.items():
            molp, *_ = inchi.split("-")
            master[molp].append(iso)

        # Build the final IsomerCollections
        mols = [IsomerCollection(isomers) for isomers in master.values()]
        self.out.send(mols)


class ExtractReference(Node):
    """Extract a single reference compound by name, smiles, or tag"""

    tags = {"chemistry", "utility"}

    inp: Input[list[IsomerCollection]] = Input()
    """Molecule inputs"""

    out: Output[Isomer] = Output()
    """Extracted reference"""

    tag: Parameter[tuple[str, ValidRDKitTagType]] = Parameter(optional=True)
    """Tag and value to use for extraction"""

    iso_name: Parameter[str] = Parameter(optional=True)
    """Name of the compound to extract"""

    def run(self) -> None:
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                if self.tag.is_set:
                    tag, val = self.tag.value
                    if iso.get_tag(tag, "") == val:
                        break
                elif self.iso_name.is_set and iso.name == self.iso_name.value:
                    break
        else:  # no break
            if self.tag.is_set:
                tag, val = self.tag.value
                msg = f"{tag} = {val}" 
            else:
                msg = f"{self.iso_name.value}"
            self.logger.warning("No reference with %s found, sending last isomer", msg)
        self.out.send(iso)


class AggregateScores(Node):
    """Create a new score by aggregating over multiple scores"""

    tags = {"chemistry", "utility", "aggregation"}

    inp: Input[list[IsomerCollection]] = Input()
    """Multiple scored molecule inputs"""

    out: Output[list[IsomerCollection]] = Output()
    """Combined scores"""

    patterns: Parameter[dict[str, str]] = Parameter()
    """
    A mapping of new-tag names to regular expressions of tags to aggregate.
    For example, ``{'foo': r'.*score.*'}`` will aggregate the values of all
    tags containing the string 'score' and assign the result to a new tag
    named 'foo'.

    """

    aggregator: Parameter[Literal["min", "max"]] = Parameter(default="min")
    """Aggregation function to use for picking the best conformer score"""

    AGGREGATORS: dict[str, Callable[[Iterable[float]], float]] = {
        "min": min,
        "max": max,
    }

    def run(self) -> None:
        compiled = {new: re.compile(pat) for new, pat in self.patterns.value.items()}
        agg = self.AGGREGATORS[self.aggregator.value]
        mols = self.inp.receive()
        for mol in mols:
            for iso in mol.molecules:
                for new, pat in compiled.items():
                    iso.add_score(
                        new,
                        agg(score for name, score in iso.scores.items() if re.match(pat, name)),
                        agg=self.aggregator.value,
                    )
        self.out.send(mols)


class LibraryFromCSV(Node):
    """
    convert a csv file into an isomer collection with columns added as tags

    """
    tags = {"chemistry", "utility", "loading"}

    inp: Input[Annotated[Path, Suffix("csv")]] = Input()
    """csv file with the molecules as SMILES in a column"""

    out: Output[list[IsomerCollection]] = Output()
    """List of molecules with single isomer and conformer after input"""

    smiles_column: Parameter[str] = Parameter(default="SMILES")
    """Name of column with structures, default is SMILES"""

    name_column: Parameter[str] = Parameter(optional=True)
    """Name of column to use as a name """

    score_column: Parameter[str] = Parameter(optional=True)
    """Name of column to use as top-level scores """

    def run(self) -> None:
        input_file = self.inp.receive()
        smiles_column = self.smiles_column.value
        if self.name_column.is_set:
            name_column = self.name_column.value
        else:
            name_column = None
        if self.score_column.is_set:
            score_column = self.score_column.value
        else:
            score_column = None

        self.logger.info("Reading %s" % str(input_file.as_posix()))

        df = pd.read_csv(input_file.as_posix())

        self.logger.info("Read %i rows and %i columns" % (df.shape[0], df.shape[1]))

        if smiles_column not in df.columns:
            raise ValueError(f"smiles column {smiles_column} not in columns {df.columns}")

        if name_column and name_column not in df.columns:
            raise ValueError(f"name column {name_column} not in columns {df.columns}")

        if score_column and score_column not in df.columns:
            raise ValueError(f"name column {score_column} not in columns {df.columns}")

        skipped_rows = []
        isomer_collections = []

        for row_ind, row in df.iterrows():
            try:
                isom = Isomer.from_smiles(row[smiles_column])

                if name_column in df.columns:
                    isom.name = row[name_column]
                else:
                    isom.name = input_file.stem + "entry-" + str(row_ind)
                for col in df.columns:
                    if score_column and col == score_column:
                        isom.add_score(score_column, row[col])
                    isom.set_tag(col, row[col])
                ic = IsomerCollection([isom])
                isomer_collections.append(ic)

            except (ChemistryException, TypeError):
                self.logger.debug(
                    f"failed read at line {int(row_ind)} with smiles {row[smiles_column]}"
                )
                skipped_rows.append(row_ind)

        self.logger.info("loaded %i rows" % len(isomer_collections))
        if len(skipped_rows):
            self.logger.info("skipped %i rows due to read fails" % len(skipped_rows))

        self.out.send(isomer_collections)


class SaveCSV(Node):
    """Save a library to a CSV file"""

    inp: Input[list[IsomerCollection]] = Input()
    """Library input"""

    file: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter(exist_required=False)
    """Output CSV file"""

    output_tags: Parameter[list[str]] = Parameter(optional=True)
    """Tags to write out"""

    format: Parameter[Literal["isomers", "collections", "livedesign"]] = Parameter(default="isomers")
    """
    How to write the CSV: `"isomers"` will write each isomer individually,
    `"collections"` will only write the first the isomer representing the
    whole collection, and `"livedesign"` writes a CSV in a format specifically
    for Schrodinger LiveDesign integration.

    """

    def run(self) -> None:
        mols = self.inp.receive()
        tags = (
            set(self.output_tags.value)
            if self.output_tags.is_set
            else set(itertools.chain(*(iso.tags.keys() for mol in mols for iso in mol.molecules)))
        )
        with self.file.filepath.open("w") as out:
            writer = csv.writer(out, delimiter=",")
            if self.format.value in ("isomers", "collections"):
                writer.writerow(["smiles", *tags])
            else:
                writer.writerow(["Corporate ID", *tags])

            for mol in mols:
                if self.format.value == "collections":
                    smiles = mol.smiles or mol.molecules[0].to_smiles(remove_h=True)
                    all_fields: dict[str, Any] = {tag: None for tag in tags}

                    # The first isomer should overwrite all others
                    for iso in reversed(mol.molecules):
                        for tag in tags:
                            val = iso.get_tag(tag, "")
                            if all_fields[tag] is None:
                                all_fields[tag] = val

                    writer.writerow([smiles, *all_fields.values()])
                    continue

                if self.format.value == "isomers":
                    for iso in mol.molecules:
                        fields = (iso.get_tag(tag, "") for tag in tags)
                        writer.writerow(
                            itertools.chain([mol.smiles or iso.to_smiles(remove_h=True)], fields)
                        )
                    continue

                if self.format.value == "livedesign":
                    for iso in mol.molecules:
                        field_vals = [iso.get_tag(tag, "") for tag in tags]
                        writer.writerow([iso.get_tag("ID"), *field_vals])


class BatchSaveCSV(Node):
    """Save library batches to a CSV file"""

    inp: Input[list[IsomerCollection]] = Input()
    """Library input"""

    file: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter(exist_required=False)
    """Output CSV file"""

    output_tags: Parameter[list[str]] = Parameter(optional=True)
    """Tags to write out"""

    n_batches: Parameter[int] = Parameter()
    """Number of batches to expect"""

    @staticmethod
    def _write_batch(writer: Any, mols: list[IsomerCollection], tags: list[str]) -> None:
        for mol in mols:
            for iso in mol.molecules:
                # We use some internal RDKit functionality here for performance reasons, we
                # can get away with it because we don't need any kind of type conversion as
                # we're immediately converting everything to a string anyway
                iso_tag_names = set(iso._molecule.GetPropNames())
                fields = (
                    iso._molecule.GetProp(tag) if tag in iso_tag_names else None for tag in tags
                )
                writer.writerow(
                    itertools.chain([mol.smiles or iso.to_smiles(remove_h=True)], fields)
                )

    def run(self) -> None:
        mols = self.inp.receive()
        if self.output_tags.is_set:
            tags = self.output_tags.value
        else:
            tags = list(
                set(itertools.chain(*(iso.tags.keys() for mol in mols for iso in mol.molecules)))
            )

        with self.file.filepath.open("a") as out:
            writer = csv.writer(out, delimiter=",")
            writer.writerow(["smiles", *tags])
            self._write_batch(writer, mols, tags)

        for batch_idx in range(self.n_batches.value - 1):
            self.logger.info("Waiting for batch %s", batch_idx + 1)
            mols = self.inp.receive()
            with self.file.filepath.open("a") as out:
                writer = csv.writer(out, delimiter=",")
                self.logger.info("Writing batch %s to %s", batch_idx, self.file.filepath.as_posix())
                self._write_batch(writer, mols, tags)


@pytest.fixture
def smiles1() -> str:
    return "Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"


@pytest.fixture
def smiles2() -> str:
    return "Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3"


class TestSuiteMol:
    def test_Smiles2Molecules(self, smiles1: str, test_config: Config) -> None:
        rig = TestRig(Smiles2Molecules, config=test_config)
        smiles = [smiles1]
        res = rig.setup_run(
            inputs={"inp": [smiles]}, parameters={"n_conformers": 2, "n_isomers": 2}
        )
        raw = res["out"].get()
        assert raw is not None
        mol = raw[0]
        assert mol.n_isomers <= 2
        assert not mol.scored
        assert mol.molecules[0].n_conformers == 2
        assert mol.molecules[0].charge == 0
        assert mol.molecules[0].n_atoms == 44

    def test_SaveMolecule(self, smiles1: str, tmp_path: Path, test_config: Config) -> None:
        rig = TestRig(SaveMolecule, config=test_config)
        mol = IsomerCollection.from_smiles(smiles1)
        rig.setup_run(inputs={"inp": mol}, parameters={"path": tmp_path / "file.sdf"})
        assert (tmp_path / "file.sdf").exists()

    def test_LoadSmiles(self, smiles1: str, shared_datadir: Path, test_config: Config) -> None:
        rig = TestRig(LoadSmiles, config=test_config)
        res = rig.setup_run(parameters={"path": shared_datadir / "test.smi"})
        mol = res["out"].get()
        assert mol is not None
        assert len(mol) == 1
        assert mol[0] == smiles1

    def test_LoadSingleRow(self, shared_datadir: Path, test_config: Config) -> None:
        rig = TestRig(LoadSingleRow, config=test_config)
        res = rig.setup_run(
            parameters={
                "path": shared_datadir / "test.smi",
                "index": 0,
                "read_csv_parameters": {"index_col": 0, "header": "infer"},
            }
        )
        line = res["out"].get()
        assert line is not None
        assert line.shape == (4,)
        assert line["product"] == "CN(C)c1ncc(-c2ccccc2)s1"

    def test_LoadLibrary(self, shared_datadir: Path, test_config: Config) -> None:
        rig = TestRig(LoadLibrary, config=test_config)
        res = rig.setup_run(parameters={"path": shared_datadir / "1UYD_ligands.sdf"})
        mols = res["out"].get()
        assert mols is not None
        assert len(mols) == 3

    def test_SaveSingleLibrary(self, smiles1: str, smiles2: str, tmp_path: Path, test_config: Config) -> None:
        rig = TestRig(SaveSingleLibrary, config=test_config)
        mols = [
            IsomerCollection.from_smiles(smiles1),
            IsomerCollection.from_smiles(smiles2),
        ]
        base = tmp_path / "mols.sdf"
        rig.setup_run(inputs={"inp": [mols]}, parameters={"file": base})
        assert base.exists()
        for orig, load in zip(mols, load_sdf_library(base)):
            assert orig.inchi == load.inchi
            assert orig.smiles == load.smiles

    def test_SaveSingleLibrary_inc(self, smiles1: str, smiles2: str, tmp_path: Path, test_config: Config) -> None:
        rig = TestRig(SaveSingleLibrary, config=test_config)
        mols = [
            IsomerCollection.from_smiles(smiles1),
            IsomerCollection.from_smiles(smiles2),
        ]
        base = tmp_path / "mols.sdf"
        base.touch()
        new = base.parent / f"{base.stem}-1.sdf"
        new.touch()
        rig.setup_run(inputs={"inp": [mols]}, parameters={"file": base})
        new2 = base.parent / f"{base.stem}-2.sdf"
        assert new2.exists()
        for orig, load in zip(mols, load_sdf_library(new2)):
            assert orig.inchi == load.inchi
            assert orig.smiles == load.smiles

    def test_SaveLibrary(self, smiles1: str, smiles2: str, tmp_path: Path, test_config: Config) -> None:
        rig = TestRig(SaveLibrary, config=test_config)
        mols = [
            IsomerCollection.from_smiles(smiles1),
            IsomerCollection.from_smiles(smiles2),
        ]
        base = tmp_path / "mol"
        rig.setup_run(inputs={"inp": [mols]}, parameters={"base_path": base})
        assert base.with_name("mol0.sdf").exists()
        assert base.with_name("mol1.sdf").exists()

    def test_SaveIsomers(self, smiles1: str, smiles2: str, tmp_path: Path, test_config: Config) -> None:
        rig = TestRig(SaveIsomers, config=test_config)
        mols = [
            IsomerCollection.from_smiles(smiles1),
            IsomerCollection.from_smiles(smiles2),
        ]
        file = tmp_path / "mol.sdf"
        rig.setup_run(inputs={"inp": [[mols[0]]]}, parameters={"file": file})
        assert file.exists()
        rmol = load_sdf_library(file, split_strategy="none")[0]
        assert rmol.molecules[0].inchi == mols[0].molecules[0].inchi
        rig.setup_run(inputs={"inp": [[mols[1]]]}, parameters={"file": file, "append": True})
        rmols = load_sdf_library(file, split_strategy="none")
        assert rmols[0].molecules[0].inchi == mols[0].molecules[0].inchi
        assert rmols[1].molecules[0].inchi == mols[1].molecules[0].inchi

    def test_CombineMolecules(self, smiles1: str, smiles2: str, test_config: Config) -> None:
        rig = TestRig(CombineMolecules, config=test_config)
        mols1 = [
            IsomerCollection.from_smiles(smiles1),
            IsomerCollection.from_smiles(smiles2),
        ]
        mols2 = deepcopy(mols1)
        for i, mols in enumerate((mols1, mols2)):
            for mol in mols:
                mol.embed(n_conformers=4)
                for iso in mol.molecules:
                    iso.set_tag("origin", f"node-{i}")
                    iso.add_score("score", float(i) * np.array([1.0, 2.0, 3.0, 4.0]))

        res = rig.setup_run(inputs={"inp": [[mols1], [mols2]]})
        comb = res["out"].get()
        assert comb is not None
        assert len(comb[0].molecules[0].scores) == 2
        assert np.allclose(comb[0].molecules[0].scores["node-0-score"], 0.0)
        assert np.allclose(comb[0].molecules[0].scores["node-1-score"], 1.0)

    def test_ExtractReference(self, smiles1: str, smiles2: str, test_config: Config) -> None:
        rig = TestRig(ExtractReference, config=test_config)
        mols = [
            IsomerCollection.from_smiles(smiles1),
            IsomerCollection.from_smiles(smiles2),
        ]
        mols[0].molecules[0].set_tag("ID", "AZ1234")
        res = rig.setup_run(inputs={"inp": [mols]}, parameters={"tag": ("ID", "AZ1234")})
        iso = res["out"].get()
        assert iso is not None
        assert iso.get_tag("ID") == "AZ1234"

    def test_AggregateScores(self, smiles1: str, smiles2: str, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles(smiles1),
            IsomerCollection.from_smiles(smiles2),
        ]
        for mol in mols:
            mol.embed(n_conformers=4)
            for iso in mol.molecules:
                iso.add_score("score-1", 1.0 * np.array([1.0, 2.0, 3.0, 4.0]))
                iso.add_score("score-2", 2.0 * np.array([1.0, 2.0, 3.0, 4.0]))

        rig = TestRig(AggregateScores, config=test_config)
        res = rig.setup_run(inputs={"inp": [mols]}, parameters={"patterns": {"foo": ".*score"}})
        comb = res["out"].get()
        assert comb is not None
        assert len(comb[0].molecules[0].scores) == 3
        assert np.allclose(comb[0].molecules[0].scores["score-1"], 1.0)
        assert np.allclose(comb[0].molecules[0].scores["score-2"], 2.0)
        assert "foo" in comb[0].molecules[0].scores
        assert np.allclose(comb[0].molecules[0].scores["foo"], 1.0)

        rig = TestRig(AggregateScores, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [mols]}, parameters={"patterns": {"foo": ".*score"}, "aggregator": "max"}
        )
        comb = res["out"].get()
        assert comb is not None
        assert len(comb[0].molecules[0].scores) == 3
        assert np.allclose(comb[0].molecules[0].scores["score-1"], 1.0)
        assert np.allclose(comb[0].molecules[0].scores["score-2"], 2.0)
        assert "foo" in comb[0].molecules[0].scores
        assert np.allclose(comb[0].molecules[0].scores["foo"], 2.0)

    def test_SaveCSV(self, smiles1: str, smiles2: str, tmp_path: Path, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles(smiles1),
            IsomerCollection.from_smiles(smiles2),
        ]
        i = 0
        for mol in mols:
            mol.embed(n_conformers=4)
            for iso in mol.molecules:
                iso.set_tag("origin", "node-0")
                iso.set_tag("ID", f"AZ1234567{i}")
                iso.set_tag("score", np.array([1, 2, 3, 4]))
                iso.add_score_tag("score")
                i += 1

        path = tmp_path / "test.csv"
        rig = TestRig(SaveCSV, config=test_config)
        rig.setup_run(
            inputs={"inp": [mols]}, parameters={"file": path, "tags": ["origin", "score"]}
        )
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert {"smiles", "origin", "score"}.issubset(set(row))
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert "node-0" in row
            assert "[1. 2. 3. 4.]" in row

        rig = TestRig(SaveCSV, config=test_config)
        rig.setup_run(inputs={"inp": [mols]}, parameters={"file": path})
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert {"smiles", "origin", "score"}.issubset(set(row))
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert "node-0" in row
            assert "[1. 2. 3. 4.]" in row

        rig = TestRig(SaveCSV, config=test_config)
        rig.setup_run(
            inputs={"inp": [mols]},
            parameters={"file": path, "format": "collections", "tags": ["origin", "score"]},
        )
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert {"smiles", "origin", "score"}.issubset(set(row))
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert "node-0" in row
            assert "[1. 2. 3. 4.]" in row

        rig = TestRig(SaveCSV, config=test_config)
        rig.setup_run(inputs={"inp": [mols]}, parameters={"file": path, "format": "livedesign"})
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert {"smiles", "origin", "score", "Corporate ID"}.issubset(set(row))
            row = next(reader)
            assert "AZ12345670" in row
            assert "node-0" in row
            assert "[1. 2. 3. 4.]" in row

    def test_BatchSaveCSV(self, tmp_path: Path, test_config: Config) -> None:
        mols = [
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3cccc(c3)OC"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OC)cc3"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OCC)cc3"),
            IsomerCollection.from_smiles("Nc1ncnc(c12)n(CCCC)c(n2)Cc3ccc(OCCC)cc3"),
        ]
        for mol in mols:
            mol.embed(n_conformers=4)
            for iso in mol.molecules:
                iso.set_tag("origin", "node-0")
                iso.set_tag("score", np.array([1, 2, 3, 4]))
                iso.add_score_tag("score")

        path = tmp_path / "test.csv"
        rig = TestRig(BatchSaveCSV, config=test_config)
        rig.setup_run(
            inputs={"inp": [mols[:2], mols[2:]]},
            parameters={"file": path, "n_batches": 2, "tags": ["origin", "score"]},
        )
        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert row == ["smiles", "origin", "score"]
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert row[1] == "node-0"
            assert row[2] == "[1, 2, 3, 4]"
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert row[1] == "node-0"
            assert row[2] == "[1, 2, 3, 4]"
            row = next(reader)
            assert row[0].startswith("Nc1ncnc(c12)")
            assert row[1] == "node-0"
            assert row[2] == "[1, 2, 3, 4]"
