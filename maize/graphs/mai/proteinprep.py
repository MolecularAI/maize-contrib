from pathlib import Path
from typing import Annotated

from maize.core.graph import Graph
from maize.core.interface import Parameter, Input, Output, Suffix
from maize.steps.mai.protein.schrod_protein_splitting import (
    ProteinChainSplitter,
    LigandProteinSplitter,
)
from maize.steps.mai.protein.prepwizard import Prepwizard
from maize.steps.mai.docking.glide_grid_generation import GlideGridGenerator
from maize.steps.plumbing import Copy


class PDBToGlideGrid(Graph):
    """Prepare a HOLO PDB structure for docking with Glide.
    This includes protein preparation, missing residue/sdie chain
    generation, protonation, chain-deduplication.
    extraction of the native ligand and grid creation

    """

    # Input
    inp: Input[Annotated[Path, Suffix("pdb")]]

    # PrepWizard
    fill_loops: Parameter[bool]
    fill_side_chains: Parameter[bool]

    # Chain splitter
    chain_to_keep: Parameter[str]

    # Ligand splitter & grid generator
    ligand_asl: Parameter[str] = Parameter(default="'res.ptype \"INH \"'")

    # Returns
    ligand_out: Output[Annotated[Path, Suffix("mae")]]
    protein_out: Output[Annotated[Path, Suffix("mae")]]
    grid_out: Output[Annotated[Path, Suffix("zip")]]

    def build(self) -> None:
        prep_instance = self.add(Prepwizard)
        chainsplit_instance = self.add(ProteinChainSplitter)
        ligsplit_instance = self.add(LigandProteinSplitter)
        gridgen_instance = self.add(GlideGridGenerator)
        protcopy_instance = self.add(Copy[Annotated[Path, Suffix("mae")]], name="pcopy")

        self.connect_all(
            (prep_instance.out, chainsplit_instance.inp),
            (chainsplit_instance.out, protcopy_instance.inp),
            (protcopy_instance.out, gridgen_instance.inp),
            (protcopy_instance.out, ligsplit_instance.inp),
        )

        self.map_port(prep_instance.inp,name="inp")
        self.map_port(ligsplit_instance.out_lig, name = "ligand_out")
        self.map_port(ligsplit_instance.out_prot,name = "protein_out")
        self.map_port(gridgen_instance.out, name = "grid_out")

        self.map(prep_instance.fill_loops, prep_instance.fill_side_chains)
        self.map(chainsplit_instance.chain_to_keep)
        self.combine_parameters(
            ligsplit_instance.ligand_asl, gridgen_instance.ligand_asl, name="ligand_asl"
        )
