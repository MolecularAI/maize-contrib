"""Helper functions to run analysis of oligonucleotide simulations"""

from typing import Any, TYPE_CHECKING
from pathlib import Path
import pandas as pd

# import MDAnalysis as mda

if TYPE_CHECKING:
    from MDAnalysis import Universe
    from MDAnalysis import AtomGroup


def average_interactions(df_inp: pd.DataFrame) -> dict[str, float]:
    """Get averaged values over simulation trajectory for each interaction type"""
    import numpy as np

    dict_int_sim = {}

    # Find unique interaction list in the data
    int_found = []
    for c in df_inp.columns:
        int_found.append(c[2])
    int_unique = set(int_found)

    # Average over the interactions in the dataframe
    for interaction in int_unique:
        df_sub = df_inp.xs(f"{interaction}", level="interaction", axis=1).replace(
            {True: 1, False: 0}
        )
        df_int = df_sub.droplevel(0, axis="columns")
        summed_int = df_int.sum(axis=1)
        avg_int = np.mean(summed_int)
        dict_int_sim[interaction] = avg_int

    return dict_int_sim


def remove_redundancy(df: pd.DataFrame) -> pd.DataFrame:
    """Removing redundant (interaction) columns from the dataframe"""

    # Summing Hbond donors and acceptors into 1 column
    df["HBonds"] = df["HBDonor"] + df["HBAcceptor"]
    df["Guide_HBonds"] = df["Guide_HBDonor"] + df["Guide_HBAcceptor"]
    df["Target_HBonds"] = df["Target_HBDonor"] + df["Target_HBAcceptor"]

    # Summing Cation-Pi and Pi-cation interactions into 1 column
    df["Pi_Cation"] = df["CationPi"] + df["PiCation"]
    df["Guide_Pi_Cation"] = df["Guide_CationPi"] + df["Guide_PiCation"]
    df["Target_Pi_Cation"] = df["Target_CationPi"] + df["Target_PiCation"]

    # Summing halogen bond donors and acceptors into 1 column
    df["Halogen"] = df["XBDonor"] + df["XBAcceptor"]
    df["Guide_Halogen"] = df["Guide_XBDonor"] + df["Guide_XBAcceptor"]
    df["Target_Halogen"] = df["Target_XBDonor"] + df["Target_XBAcceptor"]

    # Summing anionic and cationic interactions into 1 column
    df["Electrostatic"] = df["Anionic"] + df["Cationic"]
    df["Guide_Electrostatic"] = df["Guide_Anionic"] + df["Guide_Cationic"]
    df["Target_Electrostatic"] = df["Target_Anionic"] + df["Target_Cationic"]

    df.drop(
        columns=[
            "HBDonor",
            "HBAcceptor",
            "CationPi",
            "PiCation",
            "XBDonor",
            "XBAcceptor",
            "Anionic",
            "Cationic",
            "Guide_HBDonor",
            "Guide_HBAcceptor",
            "Guide_CationPi",
            "Guide_PiCation",
            "Guide_XBDonor",
            "Guide_XBAcceptor",
            "Guide_Anionic",
            "Guide_Cationic",
            "Target_HBDonor",
            "Target_HBAcceptor",
            "Target_CationPi",
            "Target_PiCation",
            "Target_XBDonor",
            "Target_XBAcceptor",
            "Target_Anionic",
            "Target_Cationic",
        ],
        inplace=True,
    )

    return df


def compute_rmsf(traj: "Universe", rna_length: int) -> float:
    """Compute RMSF for RNA backbone atoms"""
    from MDAnalysis.analysis import rms, align
    import numpy as np

    # Get an averaged structure to serve as reference for RMSF calculation
    average = align.AverageStructure(
        traj,
        traj,
        select=f"(resid 1-{rna_length*2} and (name P or name O5' or name C5' or name O3' or name C3' )) ",
        ref_frame=0,
    ).run()
    ref = average.results.universe

    # Align the trajectory to the reference structure
    align.AlignTraj(
        traj,
        ref,
        select=f"(resid 1-{rna_length*2} and (name P or name O5' or name C5' or name O3' or name C3' )) ",
        in_memory=True,
    ).run()

    # Select backbone atoms
    backbone_rna = traj.select_atoms(
        f"(resid 1-{rna_length*2} and (name P or name O5' or name C5' or name O3' or name C3' ))"
    )

    # Run RMSF calculation
    rmsf_rna = rms.RMSF(backbone_rna).run()

    # Average over the trajectory
    avg_rmsf_rna = float(np.mean(rmsf_rna.results["rmsf"]))

    return avg_rmsf_rna


def contacts_with_ions(
    u: "Universe", group_a: "AtomGroup", group_b: "AtomGroup", radius: float
) -> float:
    """Count number of sodium ions in contact with the molecule"""
    from MDAnalysis.analysis import contacts
    import numpy as np

    contact_list = []
    for ts in u.trajectory:
        # calculate distances between group_a and group_b
        dist = contacts.distance_array(group_a.positions, group_b.positions)
        # determine which distances <= radius
        n_contacts = contacts.contact_matrix(dist, radius).sum()
        contact_list.append(n_contacts)

    contact_avg = float(np.mean(contact_list))

    return contact_avg


def group_finder(ndx_file: Path, group_name: str) -> int:
    """Finds the index group number of group given by group_name"""

    group_no = 0
    with open(ndx_file, "r") as file:
        for line in file:
            if "[ " in line:
                group_no += 1
                if group_name in line:
                    return group_no - 1

    raise KeyError(f"Index group with the name {group_name} not found")
