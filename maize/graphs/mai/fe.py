"""Free energy method subgraphs and workflows"""
from pathlib import Path
from typing import Annotated, Literal

import numpy as np

from maize.core.graph import Graph
from maize.core.interface import Input, Output, Parameter, FileParameter, Suffix, Flag
from maize.steps.io import LoadFile, LogResult
from maize.steps.mai.cheminformatics import (
    RMSD,
    SetScoreTag,
    SetTag,
    BestIsomerFilter,
    TagSorter,
    SortByTag,
    TagIndex,
    ChargeFilter,
)
from maize.steps.mai.cheminformatics.taggers import SetPrimaryScore
from maize.steps.mai.md.ofe import DynamicReference, MappingType
from maize.steps.mai.molecule.mol import SaveIsomers
from maize.steps.plumbing import Copy, MergeLists
from maize.utilities.execution import JobResourceConfig

from maize.steps.mai.docking import AutoDockGPU, GNINA
from maize.steps.mai.docking.gnina import CNNScoreType
from maize.steps.mai.md import (
    OpenRFE,
    MakeAbsolute,
    MakeAbsoluteMappingScore,
)
from maize.steps.mai.molecule import Mol2Isomers, LoadMolecule
from maize.utilities.chem import Isomer, IsomerCollection


class FEPMapping(Graph):
    """A subgraph representing an FEP mapping scoring component"""

    inp: Input[list[IsomerCollection]]
    """Molecule input"""

    out: Output[list[IsomerCollection]]
    """Scored molecule output"""

    reference: FileParameter[Annotated[Path, Suffix("sdf")]]
    """Reference molecule for RMSD and ABFE calculation"""

    protein: FileParameter[Annotated[Path, Suffix("pdb")]]
    """Target protein structure"""

    exhaustiveness: Parameter[int]
    """Number of MC chains to use for conformational sampling"""

    def build(self) -> None:
        load = self.add(LoadMolecule)
        load_prot = self.add(LoadFile[Annotated[Path, Suffix("pdb")]])
        dock = self.add(GNINA)
        charge = self.add(ChargeFilter)
        best = self.add(BestIsomerFilter)
        flat = self.add(Mol2Isomers)
        fep = self.add(OpenRFE)
        copy = self.add(Copy[list[IsomerCollection]])
        copy_ref = self.add(Copy[Isomer], name="copy-ref")
        join = self.add(MakeAbsoluteMappingScore)
        prim = self.add(SetPrimaryScore)

        join.inp_ref.cached = True
        fep.inp_ref.cached = True
        fep.mapping_score_only.set(True)
        fep.mapping.set("minimal")
        prim.tag.set("mapper")

        self.connect_all(
            (load_prot.out, fep.inp_protein),
            (dock.out, charge.inp),
            (charge.out, best.inp),
            (best.out, copy.inp),
            (copy.out, flat.inp),
            (copy.out, join.inp_mols),
        )
        self.connect_all(
            (load.out, copy_ref.inp),
            (copy_ref.out, join.inp_ref),
            (copy_ref.out, fep.inp_ref),
            (copy_ref.out, charge.inp_ref),
            (copy_ref.out, dock.inp_ref),
        )
        self.connect_all((flat.out, fep.inp), (fep.out, join.inp), (join.out, prim.inp))

        self.map(dock.exhaustiveness)

        self.combine_parameters(load_prot.file, dock.receptor, name="protein")
        self.combine_parameters(load.path, name="reference")
        self.inp = self.map_port(dock.inp)
        self.out = self.map_port(prim.out)


class _FEP(Graph, register=False):
    """A subgraph representing an FEP scoring component"""

    backend: Literal["autodock", "gnina"]

    inp: Input[list[IsomerCollection]]
    """Molecule input"""

    out: Output[list[IsomerCollection]]
    """Scored molecule output"""

    equilibration_length: Parameter[int]
    """Length for equilibration simulation in ps"""

    production_length: Parameter[int]
    """Length for production simulation in ps"""

    early_termination: Flag
    """Whether to terminate calculation early if MBAR error is smaller than 0.12 kcal/mol"""

    n_repeats: Parameter[int]
    """Number of repeats to run"""

    n_replicas: Parameter[int]
    """Number of replicas to use"""

    n_lambda: Parameter[int]
    """Number of lambda windows to use"""

    reference: FileParameter[Annotated[Path, Suffix("sdf")]]
    """Reference molecule for RMSD and ABFE calculation"""

    cofactor: FileParameter[Annotated[Path, Suffix("sdf")]]
    """Optional cofactors"""

    ref_score: Parameter[float]
    """Reference ABFE"""

    protein: FileParameter[Annotated[Path, Suffix("pdb")]]
    """Target protein structure"""

    dump_to: FileParameter[Path]
    """Folder to dump FEP data to"""

    batch_options: Parameter[JobResourceConfig]
    """FEP batch options"""

    mapping: Parameter[MappingType]
    """Type of network to use for mapping"""

    mapping_backend: Parameter[Literal["lomap", "kartograf"]]
    """The mapping backend to use"""

    network: FileParameter[Annotated[Path, Suffix("edge")]]
    """An optional alternative FEPMapper atom mapping file, use ``mapping = "custom"``"""

    ref_pool: FileParameter[Annotated[Path, Suffix("sdf")]]
    """Library of existing molecules"""

    ref_score_tag: Parameter[str] = Parameter()
    """The name of the score tag in the reference"""

    target_charge: Parameter[list[int]]
    """Only isomers with this total charge pass filter"""

    trial: Flag
    """Use fake data for testing"""

    def build(self) -> None:
        load = self.add(LoadMolecule)
        load_cof = self.add(LoadMolecule, name="load-cofactor")
        load_prot = self.add(LoadFile[Annotated[Path, Suffix("pdb")]])
        indx = self.add(TagIndex)
        rmsd = self.add(RMSD)
        charge = self.add(ChargeFilter)
        best = self.add(BestIsomerFilter)
        divi = self.add(TagSorter)
        tagp = self.add(SetTag)
        score = self.add(SetScoreTag, name="score-bad")
        dynref = self.add(DynamicReference)
        flat = self.add(Mol2Isomers)
        copy_inp = self.add(Copy[list[Isomer]], name="copy-inp")
        fep = self.add(OpenRFE)
        copy = self.add(Copy[list[IsomerCollection]])
        copy_ref = self.add(Copy[Isomer], name="copy-ref")
        copy_fep = self.add(Copy[Isomer], name="copy-fep")
        join = self.add(MakeAbsolute)
        copy_all = self.add(Copy[list[IsomerCollection]], name="copy-all")
        save = self.add(SaveIsomers)
        merge = self.add(MergeLists[IsomerCollection])
        sort = self.add(SortByTag)

        dock: GNINA | AutoDockGPU
        if self.backend == "autodock":
            dock = self.add(AutoDockGPU)
            log = self.add(LogResult)
            self.connect(dock.out_scores, log.inp)
            self.map(dock.grid_file)
            self.combine_parameters(load_prot.file, name="protein")
        elif self.backend == "gnina":
            dock = self.add(GNINA)
            self.connect(copy_ref.out, dock.inp_ref)
            self.combine_parameters(load_prot.file, dock.receptor, name="protein")
            self.map(dock.exhaustiveness, dock.cnn_scoring)

        join.inp_ref.cached = True
        fep.inp_ref.cached = True
        rmsd.inp_ref.cached = True
        load_cof.path.optional = True
        fep.n_repeats.set(1)
        fep.trial.set(False)
        fep.mapping.set("minimal")
        fep.platform.set("CUDA")
        tagp.tags.set({"penalty": np.nan})
        score.tag.set("penalty")
        best.score_tag.set("rmsd")
        indx.tag.set("idx-fep")
        sort.tag.set("idx-fep")
        save.append.set(True)

        self.connect_all(
            (load_prot.out, fep.inp_protein),
            (dock.out, indx.inp),
            (indx.out, rmsd.inp),
            (rmsd.out, charge.inp),
            (charge.out, best.inp),
            (best.out, divi.inp),
        )
        self.connect_all(
            (divi.out, copy.inp),
            (divi.out, tagp.inp),
            (tagp.out, score.inp),
            (score.out, merge.inp),
            (copy.out, flat.inp),
            (copy.out, join.inp_mols),
        )
        self.connect_all(
            (load.out, copy_ref.inp),
            (copy_ref.out, dynref.inp_ref),
            (dynref.out, copy_fep.inp),
            (copy_fep.out, join.inp_ref),
            (copy_fep.out, fep.inp_ref),
            (copy_ref.out, rmsd.inp_ref),
        )
        self.connect_all(
            (copy_ref.out, charge.inp_ref),
            (flat.out, copy_inp.inp),
            (copy_inp.out, fep.inp),
            (copy_inp.out, dynref.inp),
            (fep.out, join.inp),
            (join.out, copy_all.inp),
        )
        self.connect_all(
            (copy_all.out, merge.inp),
            (copy_all.out, save.inp),
            (merge.out, sort.inp),
            (load_cof.out, fep.inp_cofactor),
        )

        self.map(
            fep.equilibration_length,
            fep.production_length,
            fep.dump_to,
            fep.batch_options,
            fep.trial,
            fep.n_repeats,
            fep.n_lambda,
            fep.n_replicas,
            fep.network,
            fep.early_termination,
            charge.target_charge,
            join.ref_score,
        )

        self.combine_parameters(load.tag, name="ref_score_tag")
        self.combine_parameters(fep.mapping, dynref.mapping, name="mapping")
        self.combine_parameters(fep.mapping_backend, dynref.mapping_backend, name="mapping_backend")
        self.combine_parameters(
            dynref.pool, save.file, name="ref_pool", default=Path("ref-pool.sdf")
        )
        self.combine_parameters(load.path, name="reference")
        self.combine_parameters(load_cof.path, name="cofactor")
        self.combine_parameters(
            divi.sorter,
            name="rmsd_cutoff",
            default=2.0,
            hook=lambda x: [f"rmsd < {x:4.2f}", f"rmsd >= {x:4.2f}"],
        )
        self.cofactor.optional = True
        self.inp = self.map_port(dock.inp)
        self.out = self.map_port(sort.out)


class FEP(_FEP):
    """FEP using AutoDock for starting structure generation"""

    grid_file: FileParameter[Annotated[Path, Suffix("fld")]]
    """Grid file for docking"""

    backend = "autodock"


class FEPGNINA(_FEP):
    """FEP using GNINA for starting structure generation"""

    exhaustiveness: Parameter[int]
    """Number of MC chains to use for conformational sampling"""

    cnn_scoring: Parameter[CNNScoreType] = Parameter(default="rescore")
    """CNN scoring method to use"""

    backend = "gnina"
