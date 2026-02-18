import os
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Literal, Sequence, cast, TYPE_CHECKING
import pickle, base64
from rdkit import DataStructs
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, Flag
from maize.utilities.utils import serialize, deserialize
from maize.utilities.chem import (
    Isomer,
    IsomerCollection,
    save_sdf_library,
    to_explicit_bitvect,
)
from maize.utilities.testing import TestRig
from maize.utilities.io import Config
import prolif


class construct_bitvect(Node):
    """
    Node to construct an ExplicitBitVect from a list of boolean values.

    This node receives a list of boolean values as input and constructs an RDKit
    ExplicitBitVect object, which is then sent as output.
    """

    inp: Input[list[bool]] = Input()
    """Input list of boolean values"""

    out: Output[ExplicitBitVect] = Output()
    """Output ExplicitBitVect"""

    def run(self) -> None:
        bits = self.inp.receive()
        bv = to_explicit_bitvect(bits)

        self.out.send(bv)


class custom_ifp(Node):
    """
    Node to construct a custom interaction fingerprint (ExplicitBitVect) based on specified residues and interaction types.

    This node generates an ExplicitBitVect where each bit represents a specific residue-interaction type pair.
    Bits are set according to the provided mapping of residues to their active interaction types.
    """

    residues: Parameter[list[str]] = Parameter()
    """List of residue identifiers."""

    interaction_type: Parameter[list[str]] = Parameter()
    """List of interaction types."""

    onbit: Parameter[dict[str, list[str]]] = Parameter()
    """Dictionary mapping residue identifiers to lists of active interaction types."""

    out: Output[ExplicitBitVect] = Output()
    """Output ExplicitBitVect representing the custom fingerprint."""

    def run(self) -> None:
        residues = self.residues.value
        interaction_types = self.interaction_type.value
        on_bits = self.onbit.value

        if residues is None or interaction_types is None or on_bits is None:
            self.logger.error("Residues, interaction types, and on bits must be provided.")
            raise ValueError("Residues, interaction types, and on bits must be provided.")

        num_residues = len(residues)
        num_types = len(interaction_types)
        fp = ExplicitBitVect(num_residues * num_types)

        for i, res in enumerate(residues):
            active_types = on_bits.get(res, [])
            for j, itype in enumerate(interaction_types):
                if itype in active_types:
                    bit_idx = i * num_types + j
                    fp.SetBit(bit_idx)

        self.out.send(fp)


class IFPsimilarity(Node):
    """
    Node to calculate similarity for a list of isomer collections.

    This node takes a list of molecules (as IsomerCollection objects) and a list of reference interactions.
    For each isomer, it calculates the percentage of interactions that match the reference set and stores this value as a score on the isomer object.

    """

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    ref_ifps: Input[list[str]] = Input()
    """Reference fingerprint for Tanimoto calculation"""

    save_library: Flag = Flag(default=False)
    """Whether to save the output SDF library file to disk."""

    destination: Parameter[Path] = Parameter(default=Path("."))
    """Destination path to save the output SDF library file."""

    out: Output[list[IsomerCollection]] = Output()
    """Output list of isomer collections with updated Tanimoto similarity scores."""

    def run(self) -> None:
        mols = self.inp.receive()
        ref_tags = self.ref_ifps.value
        total_bits = len(ref_tags)

        name = Path("tanimoto")

        for i, mol in enumerate(mols):
            for j, iso in enumerate(mol.molecules):
                cnt = 0
                for tag in ref_tags:

                    val = iso.get_tag(tag, np.nan)
                    if val == True:
                        cnt += val
                        print(cnt)
                population = cnt / total_bits
                iso.set_tag("masked_tanimoto_similarity", population)
                iso.add_score("masked_tanimoto_similarity", population, agg="max")

        self.out.send(mols)
        if self.save_library.value:
            dest_path = self.destination.value
            save_sdf_library(dest_path / Path(f"{name}_raw.sdf"), mols)


class masked_tanimoto(Node):
    """
    Node to calculate masked Tanimoto similarity for a list of isomer collections.

    The masked fingerprint is constructed by setting bits only where both the reference and other
    fingerprints have bits set. The Tanimoto similarity is then computed between the reference
    fingerprint and this masked fingerprint.

    This node receives a list of molecules (as IsomerCollection objects) and a reference
    fingerprint (ExplicitBitVect). For each isomer in each collection, it computes the
    masked Tanimoto similarity between the reference fingerprint and the isomer's fingerprint,
    and stores the result as a score in the isomer object.

    """

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    ref_ifps: Input[ExplicitBitVect] = Input()
    """Reference fingerprint for masked Tanimoto calculation"""

    save_library: Flag = Flag(default=False)
    """Whether to save the output SDF library file to disk."""

    destination: Parameter[Path] = Parameter(default=Path("."))
    """Destination path to save the output SDF library file."""

    out: Output[list[IsomerCollection]] = Output()
    """Output list of isomer collections with updated masked Tanimoto similarity scores."""

    def run(self) -> None:
        mols = self.inp.receive()
        ref_bv = self.ref_ifps.value

        name = Path("masked_tanimoto")

        for i, mol in enumerate(mols):
            for j, iso in enumerate(mol.molecules):
                tag_value = mol.get_tag("ifps")
                if isinstance(tag_value, float) and np.isnan(tag_value):
                    iso.set_tag("masked_tanimoto_similarity", np.nan)
                    iso.add_score("masked_tanimoto_similarity", np.nan, agg="max")
                    self.logger.warning(
                        f"Isomer {j} in collection {i} has NaN fingerprint, save masked_tanimoto_similarity score as np.nan."
                    )
                    continue

                other_bv = deserialize(cast(str, tag_value))
                masked = ExplicitBitVect(ref_bv.GetNumBits())
                for k in range(ref_bv.GetNumBits()):
                    if ref_bv.GetBit(k):  # Only preserve positions where ref_bv is ON
                        if other_bv.GetBit(k):
                            masked.SetBit(k)

                # Compute similarity between ref and masked other
                masked_tanimoto_similarity = DataStructs.TanimotoSimilarity(ref_bv, masked)

                iso.set_tag("masked_tanimoto_similarity", masked_tanimoto_similarity)
                iso.add_score("masked_tanimoto_similarity", masked_tanimoto_similarity, agg="max")

        self.out.send(mols)
        if self.save_library.value:
            dest_path = self.destination.value
            save_sdf_library(dest_path / Path(f"{name}_raw.sdf"), mols)


class binaryIFP(Node):

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    save_library: Flag = Flag(default=False)
    """Whether to save the output SDF library file to disk."""

    destination: Parameter[Path] = Parameter(default=Path("."))
    """Destination path to save the output SDF library file."""

    out: Output[list[IsomerCollection]] = Output()
    """Output list of isomer collections with updated masked Tanimoto similarity scores."""

    def run(self) -> None:
        mols = self.inp.receive()
        name = Path("binaryIFP")

        for i, mol in enumerate(mols):
            for j, iso in enumerate(mol.molecules):
                try:
                    val = iso.get_tag("num_interactions")
                    if val > 0:
                        iso.set_tag("binaryIFP", 1)
                        iso.add_score("binaryIFP", 1, agg="max")
                except Exception as e:
                    val = 0
                    iso.set_tag("binaryIFP", val)
                    iso.add_score("binaryIFP", val, agg="max")

        self.out.send(mols)
        if self.save_library.value:
            dest_path = self.destination.value
            save_sdf_library(dest_path / Path(f"{name}_raw.sdf"), mols)


class PLIfps(Node):
    """
    Notes
    -----
    This Node uses ProLIF library to calculate protein ligand interaction fingerprints (LIFs).
    For questions about ProLIF, please visit its official documentation at
    https://prolif.readthedocs.io/en/stable/index.html

    Description
    ------------
    If the `residues` parameter is set, it will calculate the interaction fingerprints
    for the specified residues in the protein.
    If not set, it will calculate the interaction fingerprints for all residues in the protein within a cutoff.

    The interaction fingerprints are calculated based on various interaction types such as hydrogen bond acceptors,
    hydrogen bond donors, cationic interactions, anionic interactions, cation-pi interactions, pi-cation interactions,
    pi-stacking interactions, and hydrophobic contacts. The parameters for these interactions can be customized.

    The output includes the interaction fingerprints saved as ExplicitBitVect objects, which can be serialized and stored.
    If the `residues` parameter is set, only those residues will be considered for interaction calculations. Additionaly, the number of interactions is counted and saved as a separate score in the output
    """

    # Inputs
    inp: Input[list[IsomerCollection]] = Input()
    """List of isomer collections for calculations."""

    inp_rec: Input[Path] = Input()
    """Path to a protine pdb file."""

    out: Output[list[IsomerCollection]] = Output()
    """Output includes ifps calculated, saved as ExplicitBitVect"""

    # Options
    interaction_type: Parameter[list[str]] = Parameter(
        default=[
            "HBAcceptor",
            "HBDonor",
            "Cationic",
            "PiCation",
            "PiStacking",
        ]
    )
    """"Hydrophobic", "HBAcceptor", "HBDonor", "Cationic", "Anionic", "CationPi", "PiCation", "PiStacking", "VdWContact"""

    ftf_param: Parameter[dict[str, float | tuple[int, int]]] = Parameter(
        {"distance": 6.9, "plane_angle": (0, 90), "normal_to_centroid_angle": (0, 45)}
    )

    etf_param: Parameter[dict[str, float | tuple[int, int]]] = Parameter(
        default={
            "distance": 6.5,
            "plane_angle": (50, 90),
            "normal_to_centroid_angle": (0, 30),
            "intersect_radius": 2,
        }
    )

    count: Flag = Flag(default=False)
    """Whether to keep track of all interaction occurences or just the first one."""

    # cutoff: Parameter[float] = Parameter(default=6.0)
    # """Automatically restrict the analysis to residues within this range of the ligand."""

    metadata: Flag = Flag(default=False)
    """Whether to include metadata in the output."""

    HBAcceptor: Parameter[dict[str, Any]] = Parameter(
        default={"distance": 3.7, "DHA_angle": (120, 180)}
    )
    """Hydrogen bond acceptor parameters."""

    HBDonor: Parameter[dict[str, Any]] = Parameter(
        default={"distance": 3.7, "DHA_angle": (120, 180)}
    )
    """Hydrogen bond donor parameters."""

    Cationic: Parameter[dict[str, Any]] = Parameter(default={"distance": 5.3})
    """Cationic interaction parameters."""

    CationPi: Parameter[dict[str, Any]] = Parameter(default={"distance": 5.5, "angle": (0, 30)})
    """Cation-Pi interaction parameters."""

    PiCation: Parameter[dict[str, Any]] = Parameter(default={"distance": 5.5, "angle": (0, 30)})
    """Pi-cation interaction parameters."""

    PiStacking: Parameter[dict[str, Any]] = Parameter(
        default={
            "ftf_kwargs": {
                "distance": 6.9,
                "plane_angle": (0, 90),
                "normal_to_centroid_angle": (0, 45),
            },
            "etf_kwargs": {
                "distance": 6.5,
                "plane_angle": (50, 90),
                "normal_to_centroid_angle": (0, 30),
                "intersect_radius": 2,
            },
        }
    )
    """Pi-stacking interaction parameters."""

    residues: Parameter[list[str]] = Parameter(optional=True)
    """List of residues to consider for interaction calculations."""

    save_library: Flag = Flag(default=False)
    """Whether to save the output SDF library file to disk."""

    destination: Parameter[Path] = Parameter(default=Path("."))
    """Destination path to save the output SDF library file."""

    def run(self) -> None:
        """
        Main execution method for the Prolif node. Processes input molecules and receptor, calculates interaction fingerprints,
        and saves the results.
        """
        mols = self.inp.receive()
        receptor = self.inp_rec.value
        interaction = self.interaction_type.value
        hbAcceptor = self.HBAcceptor.value
        hbDonor = self.HBDonor.value
        cationic = self.Cationic.value
        cationPi = self.CationPi.value
        piCation = self.PiCation.value
        piStacking = self.PiStacking.value
        count = self.count.value
        # cutoff = self.cutoff.value
        metaData = self.metadata.value
        name = Path("prolif_output")

        # Run
        fp = prolif.Fingerprint(
            interaction,
            parameters={
                "HBAcceptor": hbAcceptor,
                "HBDonor": hbDonor,
                "Cationic": cationic,
                "CationPi": cationPi,
                "PiCation": piCation,
                "PiStacking": piStacking,
            },
            count=count,
            # vicinity_cutoff=cutoff,
        )

        _protein = Isomer.from_pdb(receptor)
        protein_mol = prolif.Molecule.from_rdkit(_protein._molecule)
        for mol in mols:
            for iso in mol.molecules:
                plf_iso = prolif.Molecule.from_rdkit(iso._molecule)
                res1 = prolif.residue.Residue(plf_iso)

                if self.residues.is_set:
                    residues = self.residues.value
                    bits = []
                    fingerprint_failed = False
                    for residue in residues:
                        try:
                            _bits = fp.bitvector(res1, protein_mol[residue])
                            bits.extend(_bits)
                        except (ValueError, FileNotFoundError) as err:
                            fingerprint_failed = True
                            self.logger.error(
                                "Prolif failed, returning np.nan ifps for this compound"
                            )
                            break

                    if fingerprint_failed:
                        # Save placeholder for failure
                        iso.set_tag("ifps", np.nan)
                        iso.set_tag("num_interactions", 0)
                        iso.add_score("num_interactions", 0, agg="max")
                    else:
                        explicit_bits = to_explicit_bitvect(bits)
                        num_interaction = explicit_bits.GetNumOnBits()
                        ifps = serialize(explicit_bits)
                        iso.set_tag("ifps", ifps)
                        iso.set_tag("num_interactions", num_interaction)
                        iso.add_score("num_interactions", num_interaction, agg="max")

                else:
                    try:
                        fp_single = fp.generate(plf_iso, protein_mol, metadata=metaData)
                        iso.set_tag("ifps_all", serialize(fp_single.data))
                    except (ValueError, FileNotFoundError) as err:
                        iso.set_tag("ifps_all", np.nan)
                        self.logger.warning("Prolif failed, returning NaNs for all compounds")
                        continue

        self.out.send(mols)
        if self.save_library.value:
            dest_path = self.destination.value
            save_sdf_library(dest_path / Path(f"{name}_raw.sdf"), mols)


class IFPidv(Node):
    """
    Notes
    -----
    This Node extends the PLIFps Node to store each ResidueID.interactionType as a tag (1 or 0).
    Further refinement may be needed for specific use cases or improved code clarity.
    """

    # Inputs
    inp: Input[list[IsomerCollection]] = Input()
    """List of isomer collections for calculations."""

    inp_rec: Input[Path] = Input()
    """Path to a protine pdb file."""

    out: Output[list[IsomerCollection]] = Output()
    """Output includes ifps calculated, saved as ExplicitBitVect"""

    # Options
    interaction_type: Parameter[list[str]] = Parameter(
        default=[
            "HBAcceptor",
            "HBDonor",
            "Cationic",
            "PiCation",
            "PiStacking",
        ]
    )
    """"Hydrophobic", "HBAcceptor", "HBDonor", "Cationic", "Anionic", "CationPi", "PiCation", "PiStacking", "VdWContact"""

    ftf_param: Parameter[dict[str, float | tuple[int, int]]] = Parameter(
        {"distance": 6.9, "plane_angle": (0, 90), "normal_to_centroid_angle": (0, 45)}
    )

    etf_param: Parameter[dict[str, float | tuple[int, int]]] = Parameter(
        default={
            "distance": 6.5,
            "plane_angle": (50, 90),
            "normal_to_centroid_angle": (0, 30),
            "intersect_radius": 2,
        }
    )

    count: Flag = Flag(default=False)
    """Whether to keep track of all interaction occurences or just the first one."""

    cutoff: Parameter[float] = Parameter(default=6.0)
    """Automatically restrict the analysis to residues within this range of the ligand."""

    metadata: Flag = Flag(default=False)
    """Whether to include metadata in the output."""

    HBAcceptor: Parameter[dict[str, Any]] = Parameter(
        default={"distance": 3.7, "DHA_angle": (120, 180)}
    )
    """Hydrogen bond acceptor parameters."""

    HBDonor: Parameter[dict[str, Any]] = Parameter(
        default={"distance": 3.7, "DHA_angle": (120, 180)}
    )
    """Hydrogen bond donor parameters."""

    Cationic: Parameter[dict[str, Any]] = Parameter(default={"distance": 5.3})
    """Cationic interaction parameters."""

    CationPi: Parameter[dict[str, Any]] = Parameter(default={"distance": 5.5, "angle": (0, 30)})
    """Cation-Pi interaction parameters."""

    PiCation: Parameter[dict[str, Any]] = Parameter(default={"distance": 5.5, "angle": (0, 30)})
    """Pi-cation interaction parameters."""

    PiStacking: Parameter[dict[str, Any]] = Parameter(
        default={
            "ftf_kwargs": {
                "distance": 6.9,
                "plane_angle": (0, 90),
                "normal_to_centroid_angle": (0, 45),
            },
            "etf_kwargs": {
                "distance": 6.5,
                "plane_angle": (50, 90),
                "normal_to_centroid_angle": (0, 30),
                "intersect_radius": 2,
            },
        }
    )
    """Pi-stacking interaction parameters."""

    residues: Parameter[list[str]] = Parameter(optional=True)
    """List of residues to consider for interaction calculations."""

    save_library: Flag = Flag(default=False)
    """Whether to save the output SDF library file to disk."""

    destination: Parameter[Path] = Parameter(default=Path("."))
    """Destination path to save the output SDF library file."""

    def run(self) -> None:
        """
        Main execution method for the Prolif node. Processes input molecules and receptor, calculates interaction fingerprints,
        and saves the results.
        """
        mols = self.inp.receive()
        receptor = self.inp_rec.value
        interaction = self.interaction_type.value
        hbAcceptor = self.HBAcceptor.value
        hbDonor = self.HBDonor.value
        cationic = self.Cationic.value
        cationPi = self.CationPi.value
        piCation = self.PiCation.value
        piStacking = self.PiStacking.value
        count = self.count.value
        cutoff = self.cutoff.value
        name = Path("prolif_output")

        # Run
        fp = prolif.Fingerprint(
            interaction,
            parameters={
                "HBAcceptor": hbAcceptor,
                "HBDonor": hbDonor,
                "Cationic": cationic,
                "CationPi": cationPi,
                "PiCation": piCation,
                "PiStacking": piStacking,
            },
            count=count,
        )

        _protein = Isomer.from_pdb(receptor)
        protein_mol = prolif.Molecule.from_rdkit(_protein._molecule)
        interaction_order = list(fp.interactions.keys())

        for mol in mols:
            for iso in mol.molecules:
                plf_iso = prolif.Molecule.from_rdkit(iso._molecule)
                res1 = prolif.residue.Residue(plf_iso)

                if self.residues.is_set:
                    residues = self.residues.value
                else:
                    _residues = prolif.utils.get_residues_near_ligand(
                        plf_iso, protein_mol, cutoff=cutoff
                    )
                    formatted = [f"{r.name}{r.number}.{r.chain}" for r in _residues]
                    residues = formatted

                bits = []
                fingerprint_failed = False
                for residue in residues:
                    try:
                        _bits = fp.bitvector(res1, protein_mol[residue])
                        for itype, bit in zip(interaction_order, _bits):
                            iso.set_tag(f"{residue}.{itype}", bool(bit))
                            iso.add_score(f"{residue}.{itype}", bool(bit), agg="max")
                        bits.extend(_bits)
                    except (ValueError, FileNotFoundError) as err:
                        fingerprint_failed = True
                        self.logger.error("Prolif failed, returning np.nan ifps for this compound")
                        break

                if fingerprint_failed:
                    iso.set_tag("ifps", np.nan)
                    iso.set_tag("num_interactions", 0)
                    iso.add_score("num_interactions", 0, agg="max")
                else:
                    explicit_bits = to_explicit_bitvect(bits)
                    num_interaction = explicit_bits.GetNumOnBits()
                    ifps = serialize(explicit_bits)
                    iso.set_tag("ifps", ifps)
                    iso.set_tag("num_interactions", num_interaction)
                    iso.add_score("num_interactions", num_interaction, agg="max")

        self.out.send(mols)
        if self.save_library.value:
            dest_path = self.destination.value
            save_sdf_library(dest_path / Path(f"{name}_raw.sdf"), mols)


class IFPTypeCount(Node):
    """
    Notes
    -----
    This node counts the total number of interactions for each interaction type.
    If a list of residues is provided, only interactions involving those residues are included;
    otherwise, it counts interactions across all residues present in the input.
    For each isomer, the totals are stored as tags named num_<interactionType>.
    """

    # Inputs
    inp: Input[list[IsomerCollection]] = Input()
    """List of isomer collections for calculations."""

    # Parameters
    residues: Parameter[list[str]] = Parameter(optional=True)
    """List of residues to consider for interaction calculations."""

    interaction_type: Parameter[list[str]] = Parameter(
        default=[
            "HBAcceptor",
            "HBDonor",
            "Cationic",
            "PiCation",
            "PiStacking",
        ]
    )
    """"Hydrophobic", "HBAcceptor", "HBDonor", "Cationic", "Anionic", "CationPi", "PiCation", "PiStacking", "VdWContact"""

    save_library: Flag = Flag(default=False)
    """Whether to save the output SDF library file to disk."""

    destination: Parameter[Path] = Parameter(default=Path("."))
    """Destination path to save the output SDF library file."""

    # Output
    out: Output[list[IsomerCollection]] = Output()
    """Output"""

    def run(self) -> None:

        mols = self.inp.receive()
        interaction = self.interaction_type.value
        name = Path("prolif_output")

        if self.residues.is_set:
            residues = self.residues.value
            for mol in mols:
                for iso in mol.molecules:
                    type_count = {itype: 0 for itype in interaction}
                    for residue in residues:
                        for itype in interaction:
                            tag = f"{residue}.{itype}"
                            val = iso.get_tag(tag, np.nan)
                            if val == True:
                                type_count[itype] += 1
                    for itype in interaction:
                        iso.set_tag(f"num_{itype}", type_count[itype])
                        iso.add_score(f"num_{itype}", type_count[itype], agg="max")
        else:
            for mol in mols:
                for iso in mol.molecules:
                    type_count = {itype: 0 for itype in interaction}
                    for itype in interaction:
                        for tag_key in iso.tags.keys():
                            if tag_key.startswith("m_score__") and tag_key.endswith(f".{itype}"):
                                val = iso.get_tag(tag_key)
                                if val == True:
                                    type_count[itype] += 1
                    for itype in interaction:
                        iso.set_tag(f"num_{itype}", type_count[itype])
                        iso.add_score(f"num_{itype}", type_count[itype], agg="max")

        self.out.send(mols)
        if self.save_library.value:
            dest_path = self.destination.value
            save_sdf_library(dest_path / Path(f"{name}_raw.sdf"), mols)
