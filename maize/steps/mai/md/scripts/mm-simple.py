#!/usr/bin/env python

"""A simple molecular dynamics script"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
from openmm.app import PDBReporter, StateDataReporter, PDBFile
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform
import openmm.unit as u
from openff.toolkit import ForceField, Molecule, Topology
from openff.units import Quantity, unit as ffu
from pdbfixer import PDBFixer


BOX_OPTIONS = ["cubic", "dodecahedron", "octahedron"]
SOLVENT_OPTIONS = ["tip3p", "tip4pew", "spce", "tip5p"]


def insert_molecule_and_remove_clashes(
    topology: Topology,
    insert: Molecule,
    radius: Quantity = 1.5 * ffu.angstrom,
    keep: list[Molecule] = [],
) -> Topology:
    """
    Add a molecule to a copy of the topology, removing any clashing molecules.

    The molecule will be added to the end of the topology. A new topology is
    returned; the input topology will not be altered. All molecules that
    clash will be removed, and each removed molecule will be printed to stdout.
    Users are responsible for ensuring that no important molecules have been
    removed; the clash radius may be modified accordingly.

    Parameters
    ==========
    top
        The topology to insert a molecule into
    insert
        The molecule to insert
    radius
        Any atom within this distance of any atom in the insert is considered
        clashing.
    keep
        Keep copies of these molecules, even if they're clashing
    """
    # We'll collect the molecules for the output topology into a list
    new_top_mols = []
    # A molecule's positions in a topology are stored as its zeroth conformer
    insert_coordinates = insert.conformers[0][:, None, :]
    for molecule in topology.molecules:
        if any(keep_mol.is_isomorphic_with(molecule) for keep_mol in keep):
            new_top_mols.append(molecule)
            continue
        molecule_coordinates = molecule.conformers[0][None, :, :]
        diff_matrix = molecule_coordinates - insert_coordinates

        # np.linalg.norm doesn't work on Pint quantities 😢
        working_unit = ffu.nanometer
        distance_matrix = (
            np.linalg.norm(diff_matrix.m_as(working_unit), axis=-1) * working_unit
        )

        if distance_matrix.min() > radius:
            # This molecule is not clashing, so add it to the topology
            new_top_mols.append(molecule)
        else:
            print(f"Removed {molecule.to_smiles()} molecule")

    # Insert the ligand at the end
    new_top_mols.append(insert)

    # This pattern of assembling a topology from a list of molecules
    # ends up being much more efficient than adding each molecule
    # to a new topology one at a time
    new_top = Topology.from_molecules(new_top_mols)

    # Don't forget the box vectors!
    new_top.box_vectors = topology.box_vectors
    return new_top


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple MD script")
    parser.add_argument("--protein", type=Path, required=True, help="Protein PDB file")
    parser.add_argument("--mol", type=Path, required=True, help="Small molecule SDF input")
    parser.add_argument("--output", type=Path, required=True, help="Simulation output basename")
    parser.add_argument("--temperature", type=float, help="System temperature in K", default=298.15)
    parser.add_argument("--stepsize", type=float, help="Stepsize in picoseconds", default=0.004)
    parser.add_argument("--ion-conc", type=float, help="Ion concentration", default=0.15)
    parser.add_argument(
        "--neutralize",
        action=argparse.BooleanOptionalAction,
        type=bool,
        help="Neutralize the system",
        default=True,
    )
    parser.add_argument("--eq-length", type=int, help="Equilibration length in ps", default=100)
    parser.add_argument("--prod-length", type=int, help="Production length in ps", default=1000)
    parser.add_argument("--n-threads", type=int, help="Number of OpenMP threads to use", default=4)
    parser.add_argument(
        "--solvent", type=str, help="Water model to use", default="tip3p", choices=SOLVENT_OPTIONS
    )
    parser.add_argument(
        "--box", type=str, help="Box type to use", default="dodecahedron", choices=BOX_OPTIONS
    )
    parser.add_argument(
        "--padding", type=float, help="Minimum distance of the solute to the box edge", default=1.2
    )
    parser.add_argument(
        "--platform",
        type=str,
        help="Compute platform to use, can be one of 'CUDA', 'CPU', 'OpenCL', 'Reference'",
        default="CUDA",
    )
    args = parser.parse_args()

    platform = Platform.getPlatformByName(args.platform)
    os.environ["OMP_NUM_THREADS"] = str(args.n_threads)

    # Parse all structure files, the first will normally be the protein,
    # followed by the ligand and co-factors, but it doesn't really matter
    fixer = PDBFixer(args.protein.as_posix())
    fixer.addSolvent(padding=args.padding * u.nanometer, ionicStrength=args.ion_conc * u.molar, boxShape=args.box)
    protein_solvated = Path("prot-solv.pdb")
    with protein_solvated.open("w") as out:
        PDBFile.writeFile(fixer.topology, fixer.positions, file=out)
    top = Topology.from_pdb(protein_solvated)

    print("Parameterizing system...")
    mol = Molecule.from_file(args.mol.as_posix(), file_format="sdf")
    top = insert_molecule_and_remove_clashes(top, mol)
    ff = ForceField("openff-2.1.0.offxml", "ff14sb_off_impropers_0.0.4.offxml")
    interchange = ff.create_interchange(top)
    
    # Setup system
    integrator = LangevinMiddleIntegrator(
        args.temperature * u.kelvin, 1 / u.picosecond, args.stepsize * u.picosecond
    )
    simulation = interchange.to_openmm_simulation(integrator=integrator, platform=platform)

    print(f"System size: {simulation.topology.getNumAtoms()} atoms")
    print(f"Platform: {platform.getName()}, {os.environ['OMP_NUM_THREADS']} OpenMP threads")
    print(f"Equilibration length: {args.eq_length} ps ({args.eq_length / args.stepsize} steps)")
    print(f"Production length: {args.prod_length} ps ({args.prod_length / args.stepsize} steps)")

    # Minimize
    print("Minimizing...")
    simulation.minimizeEnergy()

    simulation.reporters.append(PDBReporter(args.output.as_posix(), 1000))
    simulation.reporters.append(StateDataReporter(sys.stdout, 1000, step=True, speed=True))

    # NVT
    print("NVT equilibration...")
    simulation.step(args.eq_length / args.stepsize)

    # NPT
    print("NPT production...")
    barostat = MonteCarloBarostat(1.0 * u.bar, args.temperature * u.kelvin)
    simulation.system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(args.prod_length / args.stepsize)
