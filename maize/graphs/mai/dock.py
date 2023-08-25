"""Docking workflows and subgraphs"""

from collections.abc import Callable
import json
from pathlib import Path
import sys
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

from maize.core.graph import Graph
from maize.core.workflow import Workflow, expose
from maize.core.interface import FileParameter, Parameter, Suffix, Input, Output
from maize.utilities.io import setup_workflow
from maize.steps.io import LoadData, LoadFile, SaveFile, Return, Void
from maize.steps.mai.misc import expose_reinvent
from maize.steps.plumbing import Accumulate, Scatter
from maize.steps.mai.molecule import (
    Gypsum,
    SaveMolecule,
    SaveScores,
    LoadSmiles,
    SaveLibrary,
    LoadMolecule,
    LoadLibrary,
)
from maize.steps.mai.docking import Vina, PrepareGrid, AutoDockGPU, PreparePDBQT, VinaScore
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
