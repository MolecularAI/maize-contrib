from pathlib import Path
from typing import Annotated, Literal

from maize.core.graph import Graph
from maize.core.interface import Parameter, Flag, Input, Output, Suffix

from maize.utilities.chem import IsomerCollection
from maize.steps.mai.docking.glide import GlideConfigType
from maize.steps.mai.molecule import ToSmiles, File2Molecule
from maize.steps.plumbing import Copy
from maize.graphs.mai.proteinprep import PDBToGlideGrid
from maize.graphs.mai.dock import GlideDocking


class PDBToGlideRedock(Graph):
    """Prepare a HOLO PDB structure for docking with Glide,
    and redock the native ligand into the newly generated grid,
    with matching substructure constraints, returing score.
    This includes protein preparation, missing residue/side chain
    generation, protonation, chain-deduplication.
    extraction of the native ligand, grid creation
    and the Ligprep/Glide for redokcing.
    """

    # Input
    inp: Input[Annotated[Path, Suffix("pdb")]]

    # PrepWizard
    fill_loops: Parameter[bool]
    fill_side_chains: Parameter[bool]

    # Chain splitter
    chain_to_keep: Parameter[str]

    # Ligand splitter & grid generator
    ligand_asl: Parameter[str]

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

    # Returns
    native_ligand_out: Output[Annotated[Path, Suffix("mae")]]
    protein_out: Output[Annotated[Path, Suffix("mae")]]
    grid_out: Output[Annotated[Path, Suffix("zip")]]
    redocked_ligand_out: Output[list[IsomerCollection]]

    def build(self) -> None:
        # prep
        audotprep_instance = self.add(PDBToGlideGrid)

        # docking
        gd_instance = self.add(GlideDocking)

        # load the ligand
        ligload = self.add(File2Molecule)

        # convert ligand to smiles
        tosmiles_instance = self.add(ToSmiles)

        # copy ops
        ligcopy_instance = self.add(Copy[Annotated[Path, Suffix("mae")]], name="lcopy")
        gridcopy_instance = self.add(Copy[Annotated[Path, Suffix("zip")]], name="gcopy")

        self.connect_all(
            (audotprep_instance.grid_out, gridcopy_instance.inp),
            (audotprep_instance.ligand_out, ligcopy_instance.inp),
            (gridcopy_instance.out, gd_instance.inp_grid),
        )
        self.connect_all(
            (ligcopy_instance.out, gd_instance.reference_ligand_file),
            (ligcopy_instance.out, ligload.inp),
            (ligload.out, tosmiles_instance.inp),
            (tosmiles_instance.out, gd_instance.inp)
        )

        self.map_port(audotprep_instance.inp, name="inp")

        self.map_port(ligcopy_instance.out, name="native_ligand_out")
        self.map_port(audotprep_instance.protein_out, name="protein_out")
        self.map_port(gridcopy_instance.out, name="grid_out")
        self.map_port(gd_instance.out, name="redocked_ligand_out")

        self.map(gd_instance.precision, gd_instance.keywords,
                 gd_instance.epik, gd_instance.ionization,
                 gd_instance.ph, gd_instance.ph_tolerance,
                 gd_instance.max_stereo)

        self.map(audotprep_instance.fill_loops, audotprep_instance.fill_side_chains,
                 audotprep_instance.chain_to_keep)

        self.combine_parameters(audotprep_instance.ligand_asl, name="ligand_asl",default="'res.ptype \"INH \"'")
