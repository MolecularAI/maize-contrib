from pathlib import Path
import shutil
from typing import Dict, Tuple
from maize.core.node import Node
from maize.core.interface import (
    Input,
    Output,
    Parameter,
    FileParameter,
    Flag,
)
from .file_utils import (
    get_group_ndx,
    process_files,
)


class MMGPBSA(Node):
    """
    End-state free enegy calculations with GROMACS files conducted by gmx_MMPBSA.
    gmx_MMPBSA requires minimum processing on the input structure and trajectory files.

    Before running gmx_MMPBSA, you need to make sure trajectory doesn't contain PBC.
    gmx trjconv is used to convert trajtory.

    Notes
    -----
    For questions about gmx_MMPBSA, please visit its official documentation at
    https://valdes-tresanco-ms.github.io/gmx_MMPBSA/dev/gmx_MMPBSA_running/

    """

    required_callables = ["gmx", "gmx_MMPBSA"]

    # Input
    xtc_inp: Input[Dict[Tuple[int, str], Path]] = Input()
    """Final GROMACS MD trajectory"""
    tpr_inp: Input[Dict[Tuple[int, str], Path]] = Input()
    """Structure file containing the system coordinates"""
    top_inp: Input[Dict[Tuple[int, str], Path]] = Input()
    """Topology file"""

    # Other inputs
    posreProt_inp: Input[Dict[Tuple[int, str], Path]] = Input()
    """Protein position rst file"""

    posreLig_inp: Input[Dict[Tuple[int, str], Path]] = Input()
    """Ligand position rst file"""

    protitp_inp: Input[Dict[Tuple[int, str], Path]] = Input()
    """Protein .itp file"""

    ligitp_inp: Input[Dict[Tuple[int, str], Path]] = Input()
    """Ligand .itp file"""

    ndx_inp: Input[Dict[Tuple[int, str], Path]] = Input()
    """Index file"""
    # Output
    gpbsa_results_out: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy")
    """File contains results"""

    gpbsa_xtc_out: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy")
    """Fitted trajectory with no pbc"""

    # Parameter
    fit: Parameter[str] = Parameter(default="rot+trans")
    """Option to fit molecule to ref structure in the structure file.
    The default value has been tested to be a reliable option in many tests
    Please refer to the official documentaion for more options.
    """

    center: Flag = Flag(default=True)
    """Option centers the system in the box. 
    User can select the group which is used to determine the geometrical center."""

    mmpbsa_file: FileParameter[Path] = FileParameter()
    """Input file containing all the specifications regarding the type of calculation that 
    is going to be performed"""

    lig_resnm: Parameter[str] = Parameter(default="MOL")
    """The user can select the group which is used to determine the geometrical center."""

    def run(self) -> None:
        parent_workdir = self.work_dir.as_posix()
        dict_xtc = self.xtc_inp.receive()
        dict_tpr = self.tpr_inp.receive()
        dict_top = self.top_inp.receive()

        dict_posreProt = self.posreProt_inp.receive()
        dict_posreLig = self.posreLig_inp.receive()
        dict_ligitp = self.ligitp_inp.receive()
        dict_protitp = self.protitp_inp.receive()

        dict_ndx = self.ndx_inp.receive()
        ligs_grp_ndx = get_group_ndx(dict_ndx, self.lig_resnm.value)

        gpbsa_f = self.mmpbsa_file.value
        gpbsa_tp = self.mmpbsa_file.value.stem.capitalize()
        gpbsa_commands = []
        lig_rep_gpbsa_dirs = []
        dict_gpbsa_results = {}
        dict_final_xtc = {}
        for k, v in dict_xtc.items():
            xtc_f = Path(v)
            replica_num = k[0]
            ligand_name = k[1]
            idx_c = ligs_grp_ndx[ligand_name][0]
            idx_p = ligs_grp_ndx[ligand_name][1]
            idx_l = ligs_grp_ndx[ligand_name][2]

            # Get other files based on the replica_number and ligand_num of the xtc_f
            tpr_f = dict_tpr[(replica_num, ligand_name)]
            top_f = dict_top[(replica_num, ligand_name)]

            posreProt_f = dict_posreProt[(replica_num, ligand_name)]
            posreLig_f = dict_posreLig[(replica_num, ligand_name)]
            ligitp_f = dict_ligitp[(replica_num, ligand_name)]
            protitp_f = dict_protitp[(replica_num, ligand_name)]
            ndx_f = dict_ndx[(replica_num, ligand_name)]

            lig_rep_gpbsa = parent_workdir / Path(ligand_name) / f"Replica{replica_num}" / gpbsa_tp
            lig_rep_gpbsa.mkdir(parents=True, exist_ok=True)
            lig_rep_gpbsa_dirs.append(lig_rep_gpbsa)

            top_local = lig_rep_gpbsa / top_f.name
            ndx_local = lig_rep_gpbsa / ndx_f.name

            shutil.copy(top_f, top_local)
            shutil.copy(posreProt_f, lig_rep_gpbsa / posreProt_f.name)
            shutil.copy(posreLig_f, lig_rep_gpbsa / posreLig_f.name)
            shutil.copy(ligitp_f, lig_rep_gpbsa / ligitp_f.name)
            shutil.copy(protitp_f, lig_rep_gpbsa / protitp_f.name)
            shutil.copy(ndx_f, ndx_local)

            rmvpbc_xtc_whole = lig_rep_gpbsa / Path("traj_noPBC_whole.xtc")
            rmvpbc_xtc_nojump_center = lig_rep_gpbsa / Path("traj_noPBC_whole_nojump.xtc")
            fit_xtc = lig_rep_gpbsa / Path("traj_fit.xtc")
            mmpbsa_fout = lig_rep_gpbsa / Path("FINAL_RESULTS_MMPBSA.dat")

            command_nopbc_whole = (
                f"{self.runnable['gmx']} trjconv "
                f"-s {tpr_f.as_posix()} "
                f"-f {xtc_f.as_posix()} "
                f"-o {rmvpbc_xtc_whole.as_posix()} "
                f"-pbc whole "
                f"-n {ndx_local.as_posix()} "
            )

            run_nopbc_whole = self.run_command(
                command_nopbc_whole,
                command_input="0" + "\n" + "q" + "\n",
                verbose=True,
                raise_on_failure=False,
            )
            if run_nopbc_whole.returncode != 0:
                continue

            # Remove the PBC and center
            command_nopbc_center = (
                f"{self.runnable['gmx']} trjconv "
                f"-s {tpr_f.as_posix()} "
                f"-f {rmvpbc_xtc_whole.as_posix()} "
                f"-o {rmvpbc_xtc_nojump_center.as_posix()} "
                f"-pbc nojump "
                f"-center "
                f"-n {ndx_local.as_posix()} "
            )

            run_nopbc_center = self.run_command(
                command_nopbc_center,
                command_input=f"{idx_l} \n" + "0 \n" + "q\n",
                verbose=True,
                raise_on_failure=False,
            )
            if run_nopbc_center.returncode != 0:
                continue

            # Fit molecule to reference structure
            command_fit = (
                f"{self.runnable['gmx']} trjconv "
                f"-s {tpr_f.as_posix()} "
                f"-f {rmvpbc_xtc_nojump_center.as_posix()} "
                f"-o {fit_xtc.as_posix()} "
                f"-n {ndx_local.as_posix()} "
                f"-fit {self.fit.value} "
            )
            run_fit = self.run_command(
                command_fit, command_input=f"{idx_c} \n" + "0 \n" + "q\n", verbose=True
            )
            if run_fit.returncode != 0:
                continue

            # PBSA part
            local_gpbsain = lig_rep_gpbsa / gpbsa_f.name
            shutil.copy(gpbsa_f, local_gpbsain)

            command_gmxmmpbsa = (
                f"{self.runnable['gmx_MMPBSA']} "
                f"-i {local_gpbsain.as_posix()} "  # .in file
                f"-cs {tpr_f.as_posix()} "
                f"-cg {idx_p} {idx_l} "
                f"-ci {ndx_f.as_posix()} "
                f"-ct {fit_xtc.as_posix()} "
                f"-cp {top_local.as_posix()} "
                f"-o {mmpbsa_fout.as_posix()} "  # FINAL_RESULTS_MMPBSA.dat
                f"-nogui "
            )
            gpbsa_commands.append(command_gmxmmpbsa)

        self.run_multi(
            commands=gpbsa_commands,
            working_dirs=lig_rep_gpbsa_dirs,
            verbose=True,
            raise_on_failure=False,
        )

        dict_gpbsa_results = process_files(lig_rep_gpbsa_dirs, gpbsa_tp, mmpbsa_fout.name)
        dict_final_xtc = process_files(lig_rep_gpbsa_dirs, gpbsa_tp, fit_xtc.name)

        self.gpbsa_results_out.send(dict_gpbsa_results)
        self.gpbsa_xtc_out.send(dict_final_xtc)
