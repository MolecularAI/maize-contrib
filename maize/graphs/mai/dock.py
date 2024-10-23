"""Docking workflows and subgraphs"""

import json
from pathlib import Path
import sys
from typing import Annotated, Literal

import numpy as np
from numpy.typing import NDArray

from maize.core.graph import Graph
from maize.core.workflow import Workflow, expose
from maize.core.interface import Flag, FileParameter, Parameter, Suffix, Input, Output
from maize.utilities.io import setup_workflow
from maize.steps.io import LoadData, LoadFile, SaveFile, Return, Void
from maize.steps.mai.misc import expose_reinvent
from maize.steps.plumbing import Accumulate, Scatter, Copy
from maize.steps.mai.molecule import (
    Gypsum,
    File2Molecule,
    SaveMolecule,
    SaveScores,
    LoadSmiles,
    SaveLibrary,
    LoadMolecule,
    LoadLibrary,
    Ligprep,
    CombineMolecules,
    AggregateScores,
    SaveCSV,
)
from maize.steps.mai.docking import AutoDockGPU, Glide, PrepareGrid, PreparePDBQT, Vina, VinaScore
from maize.steps.mai.docking.glide import GlideConfigType
from maize.utilities.chem import IsomerCollection


@expose_reinvent
class DockGPU(Graph):
    """Dock multiple molecules in the form of SMILES using AutoDockGPU and return their scores."""

    inp: Input[list[str]]
    out: Output[NDArray[np.float32]]

    # Gypsum
    n_variants: Parameter[int]
    ph_range: Parameter[tuple[float, float]]
    n_jobs: Parameter[int]
    thoroughness: Parameter[int]

    # ADGPU
    grid_file: FileParameter[Path]
    heurmax: Parameter[int]

    # Mol save location
    base_path: FileParameter[Path]

    def build(self) -> None:
        gyp = self.add(Gypsum)
        adg = self.add(AutoDockGPU)
        log = self.add(SaveLibrary)

        adg.scores_only.set(False)
        adg.strict.set(False)

        self.connect_all((gyp.out, adg.inp), (adg.out, log.inp))
        self.map(
            gyp.n_variants,
            gyp.ph_range,
            gyp.n_jobs,
            gyp.thoroughness,
            adg.grid_file,
            adg.heurmax,
            log.base_path,
        )
        self.inp = self.map_port(gyp.inp)
        self.out = self.map_port(adg.out_scores)


class Docking(Graph):
    """Dock a molecule in the form of a SMILES string to a protein."""

    # Gypsum
    n_variants: Parameter[int]
    ph_range: Parameter[tuple[float, float]]

    # Autodock
    receptor: FileParameter[Path]
    search_center: Parameter[tuple[float, float, float]]
    search_range: Parameter[tuple[float, float, float]]
    n_jobs: Parameter[int]
    n_poses: Parameter[int]

    def build(self) -> None:
        gyp = self.add(Gypsum)
        adv = self.add(Vina)
        self.connect(gyp.out, adv.inp)

        self.inp = self.map_port(gyp.inp)
        self.out = self.map_port(adv.out)
        self.map(gyp.n_variants, gyp.ph_range)
        self.map(adv.receptor, adv.search_center, adv.search_range, adv.n_jobs, adv.n_poses)


class GlideDocking(Graph):
    """
    Dock a molecule in the form of a SMILES string or
    IsomerCollection to a protein using Glide/LigPrep

    """

    inp: Input[list[str] | list[IsomerCollection]]
    out: Output[list[IsomerCollection]]

    # parallelization
    n_jobs: Parameter[int]
    host: Parameter[str]
    query_interval: Parameter[int]

    # LigPrep
    epik: Flag
    ionization: Parameter[Literal[0, 1, 2]]
    ph: Parameter[float]
    ph_tolerance: Parameter[float]
    max_stereo: Parameter[int]

    # Glide
    inp_grid: Input[Annotated[Path, Suffix("zip")]]
    precision: Parameter[Literal["SP", "XP", "HTVS"]]
    keywords: Parameter[GlideConfigType]
    core_definition: Parameter[Literal["mcssmarts"]]
    max_score: Parameter[float]

    # Reference ligand loader
    reference_ligand_file: Input[Annotated[Path, Suffix("mae", "sdf")]]

    # Reference contraint loader, optional
    constraints: FileParameter[Annotated[Path, Suffix("in")]]

    def build(self) -> None:
        ligprep_instance = self.add(Ligprep)
        glide_instance = self.add(Glide)
        ligand_loader = self.add(File2Molecule)
        ligand_loader.inp.optional = True

        self.connect_all(
            (ligprep_instance.out, glide_instance.inp),
            (ligand_loader.out, glide_instance.ref_ligand),
        )

        self.map_port(ligprep_instance.inp, name="inp")
        self.map_port(ligand_loader.inp, name="reference_ligand_file")
        self.reference_ligand_file.optional = True
        self.map_port(glide_instance.out, name="out")

        self.map(
            ligprep_instance.epik,
            ligprep_instance.ionization,
            ligprep_instance.ph,
            ligprep_instance.ph_tolerance,
            ligprep_instance.max_stereo,
        )
        self.map(
            glide_instance.inp_grid,
            glide_instance.constraints,
            glide_instance.precision,
            glide_instance.keywords,
            glide_instance.core_definition,
            glide_instance.max_score,
        )
        self.combine_parameters(glide_instance.host, ligprep_instance.host, name="host")
        self.combine_parameters(glide_instance.n_jobs, ligprep_instance.n_jobs, name="n_jobs")
        self.combine_parameters(glide_instance.query_interval, ligprep_instance.query_interval, name="query_interval")


@expose
def dock_glide() -> Workflow:
    flow = Workflow(name="glide")
    load = flow.add(LoadSmiles)
    glide = flow.add(GlideDocking)
    save = flow.add(SaveLibrary)
    flow.connect(load.out, glide.inp)  # type: ignore
    flow.connect(glide.out, save.inp)
    flow.combine_parameters(load.path, name="smiles")
    flow.combine_parameters(glide.inp_grid, name="grid")
    flow.combine_parameters(glide.reference_ligand_file, name="reference")
    flow.combine_parameters(save.base_path, name="output")
    flow.map(
        glide.keywords,
        glide.precision,
        glide.ph,
        glide.ph_tolerance,
        glide.max_stereo,
        glide.epik,
        glide.ionization,
    )
    return flow


@expose
def prepare_pdbqt() -> Workflow:
    """Prepares a PDBQT file for Vina from a PDB file"""
    flow = Workflow(name="pdbqt-prepare")
    load_protein = flow.add(LoadFile[Annotated[Path, Suffix("pdb")]], name="protein")
    prep = flow.add(PreparePDBQT)
    save = flow.add(SaveFile[Annotated[Path, Suffix("pdbqt")]])

    flow.connect(load_protein.out, prep.inp)
    flow.connect(prep.out, save.inp)

    save.overwrite.set(True)
    flow.combine_parameters(load_protein.file, name="protein")
    flow.combine_parameters(save.destination, name="output")
    flow.map(prep.repairs, prep.cleanup_protein, prep.preserve_charges, prep.remove_nonstd)
    return flow


@expose
def prepare_grid() -> Workflow:
    """Prepares a grid for AutoDockGPU from a PDB file"""
    flow = Workflow(name="grid-prepare")
    load_protein = flow.add(LoadFile[Annotated[Path, Suffix("pdb")]], name="protein")
    load_ligand = flow.add(LoadMolecule, name="ligand")
    prep = flow.add(PrepareGrid)
    save = flow.add(SaveFile[Annotated[Path, Suffix("tar")]])

    flow.connect(load_protein.out, prep.inp_structure)
    flow.connect(load_ligand.out, prep.inp_ligand)
    flow.connect(prep.out, save.inp)

    save.overwrite.set(True)
    load_ligand.path.optional = True
    flow.combine_parameters(load_protein.file, name="protein")
    flow.combine_parameters(load_ligand.path, name="ligand")
    flow.combine_parameters(save.destination, name="output")
    flow.map(prep.search_center, prep.search_range)
    return flow


def dock_gpu() -> None:
    """Dock multiple SMILES codes to a target using AutoDockGPU"""
    flow = Workflow(name="vina")
    smi = flow.add(LoadData[list[str]])
    gyp = flow.add(Gypsum)
    adg = flow.add(AutoDockGPU)
    sco = flow.add(Return[NDArray[np.float32]], name="scores")
    log = flow.add(SaveLibrary)

    # Small hack to allow us to use help, but at the
    # same time allow reading in all input from stdin
    if all(flag not in sys.argv for flag in ("-h", "--help")):
        smiles = sys.stdin.readlines()
        smi.data.set(smiles)
    adg.scores_only.set(False)
    adg.strict.set(False)

    flow.connect_all(
        (smi.out, gyp.inp), (gyp.out, adg.inp), (adg.out, log.inp), (adg.out_scores, sco.inp)
    )
    flow.map(
        gyp.n_variants,
        gyp.ph_range,
        gyp.n_jobs,
        gyp.thoroughness,
        adg.grid_file,
        adg.heurmax,
        log.base_path,
    )
    setup_workflow(flow)
    # 1 is stdout, 2 is stderr
    if (scores := sco.get()) is not None:
        print(json.dumps(list(scores)))


@expose
def dock_multi(n_grids: int = 2) -> Workflow:
    """Dock multiple SMILES to multiple targets using AutoDockGPU"""
    flow = Workflow(name="multi-grid")
    smi = flow.add(LoadSmiles)
    gyp = flow.add(Gypsum)
    copy = flow.add(Copy[list[IsomerCollection]])
    adgs = [flow.add(AutoDockGPU, name=f"adg-{i}") for i in range(n_grids)]
    void = flow.add(Void)
    comb = flow.add(CombineMolecules)
    agg = flow.add(AggregateScores)
    log = flow.add(SaveCSV)

    flow.connect_all((smi.out, gyp.inp), (gyp.out, copy.inp))
    for i, adg in enumerate(adgs):
        adg.scores_only.set(False)
        adg.strict.set(False)
        flow.combine_parameters(adg.grid_file, name=f"grid_{i}")
        flow.connect_all((copy.out, adg.inp), (adg.out_scores, void.inp), (adg.out, comb.inp))
    flow.connect(comb.out, agg.inp)
    flow.connect(agg.out, log.inp)

    agg.patterns.set({"score": ".*"})

    flow.combine_parameters(smi.path, name="smiles")
    flow.combine_parameters(log.file, name="output")
    flow.map(
        gyp.n_variants,
        gyp.ph_range,
        gyp.n_jobs,
        gyp.thoroughness,
    )
    return flow


@expose
def score_only() -> Workflow:
    """Score a library of molecules from an SDF file using Vina"""
    flow = Workflow(name="score")
    lib = flow.add(LoadLibrary)
    vin = flow.add(VinaScore)
    log = flow.add(Void)
    save = flow.add(SaveScores)

    flow.connect_all(
        (lib.out, vin.inp),
        (vin.out, log.inp),
        (vin.out_scores, save.inp),
    )

    flow.combine_parameters(save.path, name="output")
    flow.map(vin.receptor, lib.path)

    return flow


@expose
def dock() -> Workflow:
    """Dock a library of SMILES codes to a target"""
    flow = Workflow(name="vina")
    smiles = flow.add(LoadSmiles)
    dock = flow.add(Docking)
    save = flow.add(SaveLibrary)

    flow.connect(smiles.out, dock.inp)
    flow.connect(dock.out, save.inp)
    flow.combine_parameters(smiles.path, name="smiles")
    flow.combine_parameters(save.base_path, name="output")
    flow.map(
        dock.n_variants,
        dock.ph_range,
        dock.receptor,
        dock.search_center,
        dock.search_range,
        dock.n_jobs,
        dock.n_poses,
    )
    return flow


@expose
def dock_single() -> Workflow:
    flow = Workflow(name="vina")
    smiles, pack, dock, unpack, save = flow.add_all(
        LoadData[str], Accumulate[str], Docking, Scatter[IsomerCollection], SaveMolecule
    )
    pack.n_packets.set(1)
    flow.connect_all(
        (smiles.out, pack.inp), (pack.out, dock.inp), (dock.out, unpack.inp), (unpack.out, save.inp)
    )

    flow.combine_parameters(smiles.data, name="smiles")
    flow.combine_parameters(save.path, name="output")
    flow.map(
        dock.n_variants,
        dock.ph_range,
        dock.receptor,
        dock.search_center,
        dock.search_range,
        dock.n_jobs,
        dock.n_poses,
    )
    return flow
