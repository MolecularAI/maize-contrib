from pathlib import Path
import logging
import shutil
import pytest
from collections import defaultdict
from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, FileParameter, Flag
from maize.utilities.testing import TestRig
from maize.utilities.resources import cpu_count
from typing import Any, Dict, Tuple, Literal
from .file_utils import (
    MDPFileParser,
    generate_replicas,
    get_index,
    merge_pdb,
    merge_top,
    process_files,
    extract_filename,
    convert_data_dict,
)
from maize.utilities.execution import JobResourceConfig


class MDs(Node):
    """
    MD simulations conducted by GROMACS.
    It includes pdb2gmx, editconf, solvate, genion, make_ndx, grompp and mdrun commands.
    ACPYPE is used for small molecule paramterization.

    Notes
    -----
    This Node includes essential commands to start a gmx MD run.
    For questions about gmx_MMPBSA, please visit its official documentation at
    https://manual.gromacs.org/current/index.html

    """

    required_callables = ["gmx", "acpype"]

    # Inputs
    inp: Input[Path] = Input()
    """pdb2mx input file, path to a protine pdb file"""

    ff: Parameter[str] = Parameter(default="amber03")
    """"pdb2gmx: Force field for a protein. 
    Options can be found in path/to/gromacs/top folder.
    Names of all residues in the PDB file adhere to the selected force field conventions."""

    ff_wat: Parameter[str] = Parameter(default="tip3p")
    """"pdb2gmx: Force field for solvant. 
    Options can be found in path/to/gromacs/top folder"""

    inp_lig: Input[list[Path]] = Input()
    """A list of file pathes of small molecules. Accept files in PDB and SDF formats."""

    # Outputs
    out_topol_tpr: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy")
    """Tpr file as output"""

    out_topol_top: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy")
    """Top file as output"""

    out_confout_gro: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Structure file"""

    out_ener_edr: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Structure file"""

    out_state_prev_cpt: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Previous state checkpoint file"""

    out_traj_trr: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Trajectory file"""

    out_state_cpt: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Checkpoint file"""

    out_traj_xtc: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Comporessed trajectory file"""

    out_md_log: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy")
    """MD log file"""

    out_posreProt: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Protein position rst file"""

    out_posreLig: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Ligand position rst file"""

    out_protItp: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Protein itp file"""

    out_ligItp: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Ligand itp file"""

    out_index: Output[Dict[Tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Index file"""

    # Options
    set_posreProt: Flag = Flag(default=True)
    """mdrun: Position restraint file to a protein"""

    set_posreLig: Flag = Flag(default=True)
    """mdrun: Position restraint file to a small molecule"""

    rst_file: Flag = Flag(default=True)
    """mdrun: Structure file: gro"""

    index_file: Flag = Flag(default=True)
    """mdrun: Index file"""

    replicas: Parameter[int] = Parameter(default=1)
    """generate_replicas: Number of replicas"""

    ignore_Hatoms: Flag = Flag(default=False)
    """pdb2gmx: Ignore hydrogen atoms that are in the coordinate file.
    If you wish to maintain the protonation state, you should use "False"."""

    charge_method: Parameter[Literal["bcc", "gas"]] = Parameter(default="bcc")
    """acpype: Charge methods"""

    output_format: Parameter[Literal["all", "gmx", "cns", "charmm"]] = Parameter(default="all")
    """acpype: Output file format"""

    net_charge: Parameter[int] = Parameter(default=0)
    """acpype: Net molecular charge, it tries to guess it if not not declared"""

    atom_type: Parameter[Literal["gaff", "amber", "gaff2", "amber2"]] = Parameter(default="gaff2")
    """acpype: atom_type"""

    box_type: Parameter[Literal["triclinic", "cubic", "octahedron", "dodecahedron"]] = Parameter(
        default="cubic"
    )
    """editconf: Box type"""

    distance: Parameter[float] = Parameter(default=1.2)
    """editconf: Distance between the solute and the box"""

    vdw_radius: Parameter[float] = Parameter(default=0.12)
    """editconf: Default Van der Waals radius (in nm). 
    If one can not be found in the database or if no parameters are present in the topology file"""

    solvant_model: Parameter[str] = Parameter(default="spc216")
    """solvate: Structure file for solvent. 
    Options can be found in path/to/gromcas/top folder"""

    mdp_file: FileParameter[Path] = FileParameter()
    """grompp: Grompp input file with MD parameters"""

    max_warn: Parameter[int] = Parameter(default=0)
    """grompp: Number of allowed warnings during input processing."""

    replace_with: Parameter[str] = Parameter(default="SOL")
    """geneion: Replace solvent molecules with monoatomic ions"""

    group1: Parameter[str] = Parameter(default="Protein")
    """make_mdx: Merge one group with the other one"""

    group2: Parameter[str] = Parameter(default="MOL")
    """make_mdx: Merge one group with the other one"""

    mdp_files: FileParameter[list[Path]] = FileParameter()
    """grompp: Grompp input file with MD parameters, for different MD setup"""

    new_mdp_values: Parameter[Any] = Parameter(default_factory=dict)
    """Don't use it in current version"""

    num_threads: Parameter[int] = Parameter(default=4)
    """mdrun: Total number of threads to start"""

    num_tmpi: Parameter[int] = Parameter(default=1)
    """mdrun: Number of thread-MPI ranks to start"""

    num_tomp: Parameter[int] = Parameter(default=4)
    """mdrun: Number of OpenMP threads per MPI rank to start"""

    cpt_interval: Parameter[int] = Parameter(default=15)  # default 15 min
    """mdrun: Checkpoint interval (minutes)"""

    # One extra parameter to control which file to send
    sendout_option: Parameter[list[str]] = Parameter(
        default=[
            "topol_tpr",
            "topol_top",
            "confout_gro",
            "ener_edr",
            "md_log",
            "traj_trr",
            "traj_xtc",
            "state_cpt",
            "state_prev_cpt",
        ]
    )
    """Options to send out various files"""

    # Run
    # Pdb2gmx
    def run(self) -> None:
        structure_in = self.inp.receive()
        ff_in = self.ff.value
        wat_in = self.ff_wat.value

        structure_out = Path("Protein.pdb")
        topology_out = Path("Protein.top")
        posre_out = Path("posre.itp")

        if not self.ignore_Hatoms.value:
            self.logger.debug(" pdb2gmx self.workdir() %s ", self.work_dir)
            self.logger.debug(
                " pdb2gmx structure_in.absolute().exists() %s ", structure_in.exists()
            )

            command_pdb2gmx = (
                f"{self.runnable['gmx']} pdb2gmx "
                f"-f {structure_in.as_posix()} "
                f"-water {wat_in.strip() } "
                f"-ff {ff_in.strip() } "
                f"-o {structure_out.as_posix()} "
                f"-p {topology_out.as_posix()} "
            )
        else:
            command_pdb2gmx = (
                f"{self.runnable['gmx']} pdb2gmx "
                f"-f {structure_in.as_posix()} "
                f"-water {wat_in.strip() } "
                f"-ff {ff_in.strip() } "
                f"-o {structure_out.as_posix()} "
                f"-p {topology_out.as_posix()} "
                f"-ignh "
            )
        self.run_command(command_pdb2gmx)
        self.logger.debug(
            "Protein.pdb is generated %s and saved in %s",
            structure_out.exists(),
            structure_out.absolute(),
        )

        # Acpype
        ligands = self.inp_lig.receive()
        premd_dict_posreProt: Dict[Tuple[int, str], Path] = {}
        premd_dict_posreLig: Dict[Tuple[int, str], Path] = {}
        premd_dict_protItp: Dict[Tuple[int, str], Path] = {}
        premd_dict_ligItp: Dict[Tuple[int, str], Path] = {}
        premd_dict_index: Dict[Tuple[int, str], Path] = {}
        premd_dict_top: Dict[Tuple[int, str], Path] = {}
        premd_dict_gro: Dict[Tuple[int, str], Path] = {}

        num_replicas = self.replicas.value
        parent_workdir = self.work_dir
        self.logger.info("Ligands received %s", ligands)
        self.logger.info("Number of replicas will be excuted on each ligand %s", num_replicas)

        for ligand in ligands:
            ligand_name = ligand.stem
            self.logger.info("Starting with %s", ligand)
            # Create subfolder name with Ligand_name
            lig_sub = parent_workdir / ligand_name
            self.logger.debug(
                "Making a sub-folder for ligand %s at %s", ligand_name, lig_sub.as_posix()
            )
            lig_sub.mkdir()
            if not lig_sub.exists():
                self.logger.debug("Failed to creat the subfolder")
                continue

            # Output
            # Files will be moved to subfolder
            acpype_itp = Path(f"{ligand_name}.acpype/{ligand_name}_GMX.itp")
            acpype_top = Path(f"{ligand_name}.acpype/{ligand_name}_GMX.top")
            acpype_pdb = Path(f"{ligand_name}.acpype/{ligand_name}_NEW.pdb")
            acpype_posre = Path(f"{ligand_name}.acpype/posre_{ligand_name}.itp")

            # Acpype command
            command_acpype = (
                f"{self.runnable['acpype']} "
                f"-di {ligand.as_posix()} "
                f"-c {self.charge_method.value} "
                f"-n {self.net_charge.value} "
                f"-a {self.atom_type.value} "
            )

            run_acpype = self.run_command(command_acpype, raise_on_failure=False)
            if run_acpype.returncode != 0:
                continue

            # Copy necessary files from {ligand_name}.acpype to ligand_name
            lig_acpype_itp = lig_sub / f"{ligand_name}_GMX.itp"
            lig_acpype_top = lig_sub / f"{ligand_name}_GMX.top"
            lig_acpype_pdb = lig_sub / f"{ligand_name}_NEW.pdb"
            lig_acpype_posre = lig_sub / f"posre_{ligand_name}.itp"
            shutil.copy(acpype_itp, lig_acpype_itp)
            shutil.copy(acpype_top, lig_acpype_top)
            shutil.copy(acpype_pdb, lig_acpype_pdb)
            shutil.copy(acpype_posre, lig_acpype_posre)

            # Copy protein related files to each ligand subfolder
            prot_pdb = lig_sub / "Protein.pdb"
            prot_top = lig_sub / "Protein.top"
            posreProt = lig_sub / "posre.itp"
            shutil.copy(structure_out, prot_pdb)
            shutil.copy(topology_out, prot_top)
            shutil.copy(posre_out, posreProt)

            # merge_pdb
            try:
                out_pdb_mergePDB = merge_pdb(lig_sub, prot_pdb, lig_acpype_pdb)
                self.logger.info(
                    " Wrote complex structure as %s, successed? %s",
                    out_pdb_mergePDB,
                    out_pdb_mergePDB.absolute().exists(),
                )
            except (ValueError, FileNotFoundError) as err:
                self.logger.debug(err)
                continue

            # merge_top
            try:
                out_protitp_mergeTop, out_ligitp_mergeTop, out_top_mergeTop = merge_top(
                    lig_sub, prot_top, lig_acpype_itp, lig_acpype_top
                )
                self.logger.info(
                    "Wrote protein itp file to the ligand subfolder %s", out_protitp_mergeTop
                )
                self.logger.info(
                    "Wrote ligand itp file to the ligand subfolder %s", out_ligitp_mergeTop
                )
                self.logger.info(
                    "Wrote topol.top file to the Ligand subfolder  %s", out_top_mergeTop
                )
            except (ValueError, FileNotFoundError) as err:
                self.logger.debug(err)
                continue

            # Editconf
            editconf_in = out_pdb_mergePDB
            editconf_out = lig_sub / "confout.gro"
            command_editconf = (
                f"{self.runnable['gmx']} editconf "
                f"-f {editconf_in.as_posix()} "
                f"-o {editconf_out.as_posix()} "
                f"-bt {self.box_type.value } "
                f"-d {self.distance.value } "
                f"-c "
            )
            run_editconf = self.run_command(
                command_editconf, working_dir=lig_sub, raise_on_failure=False
            )
            if run_editconf.returncode != 0:
                continue

            # Solvate
            solvate_gro_in = editconf_out
            solvate_io = lig_sub / "confout.gro"
            topology_io = out_top_mergeTop
            command_solvate = (
                f"{self.runnable['gmx']} solvate "
                f"-cp {solvate_gro_in.as_posix()} "
                f"-cs {self.solvant_model.value} "
                f"-o {solvate_io.as_posix()} "
                f"-p {topology_io.as_posix()} "
            )
            run_solvate = self.run_command(
                command_solvate, working_dir=lig_sub, raise_on_failure=False
            )
            if run_solvate.returncode != 0:
                continue

            # Grompp
            grompp_gro_in = solvate_io
            grompp_out_mdrun = lig_sub / Path("topol.tpr")

            from_mdp = self.mdp_file.filepath
            mdp_local = lig_sub / from_mdp.name
            shutil.copy(from_mdp, mdp_local)

            grompp_command = (
                f"{self.runnable['gmx']} grompp "
                f"-p {topology_io.as_posix()} "
                f"-c {grompp_gro_in.as_posix()} "
                f"-f {mdp_local.as_posix()} "
                f"-maxwarn {self.max_warn.value} "
                f"-o {grompp_out_mdrun.as_posix()} "
            )
            run_grompp = self.run_command(grompp_command, raise_on_failure=False)
            if run_grompp.returncode != 0:
                continue

            # MakeIndex
            mkndx_gro_in = grompp_gro_in
            ndx = lig_sub / Path("index.ndx")
            command_mkndx = (
                f"{self.runnable['gmx']} make_ndx "
                f"-f {mkndx_gro_in.as_posix()} "
                f"-o {ndx.as_posix()} "
            )
            run_mkndx = self.run_command(command_mkndx, command_input="q\n", raise_on_failure=False)
            if run_mkndx.returncode != 0:
                continue
            self.logger.info("Make_ndx wrote index file at %s", ndx.absolute())

            # Genion
            genion_gro = mkndx_gro_in
            genion_tpr = grompp_out_mdrun
            genion_top = topology_io
            var = self.replace_with.value

            idx_sol = get_index(ndx, var)
            self.logger.debug("idx_sol %s", idx_sol)
            command_genion = (
                f"{self.runnable['gmx']} genion "
                f"-pname NA "
                f"-nname CL "
                f"-neutral "
                f"-o {genion_gro.as_posix()} "
                f"-p {genion_top.as_posix()} "
                f"-s {genion_tpr.as_posix()} "
            )
            run_genion = self.run_command(
                command_genion,
                command_input=f"{idx_sol}" + "\n" + "q" + "\n",
                verbose=True,
                working_dir=lig_sub,
                raise_on_failure=False,
            )
            if run_genion.returncode != 0:
                continue

            # Merge_ndx
            mergendx_gro = genion_gro
            mergendx_ndx = lig_sub / Path("index.ndx")

            idx_Protein = get_index(mergendx_ndx, self.group1.value)
            idx_LIG = get_index(mergendx_ndx, self.group2.value)

            command_mergendx = (
                f"{self.runnable['gmx']} make_ndx "
                f"-f {mergendx_gro.as_posix()} "
                f"-o {mergendx_ndx.as_posix()} "
            )
            run_Merge_ndx = self.run_command(
                command_mergendx,
                command_input=f"{idx_Protein}|{idx_LIG}" + "\n" + "q" + "\n",
                verbose=True,
                raise_on_failure=False,
            )
            if run_Merge_ndx.returncode != 0:
                continue
            mdtp = mdp_local.stem.capitalize()

            a = [
                premd_dict_posreProt,
                premd_dict_posreLig,
                premd_dict_protItp,
                premd_dict_ligItp,
                premd_dict_index,
                premd_dict_top,
                premd_dict_gro,
            ]
            b = [
                posreProt,
                lig_acpype_posre,
                out_protitp_mergeTop,
                out_ligitp_mergeTop,
                mergendx_ndx,
                genion_top,
                genion_gro,
            ]
            for dict_with_replica, file in zip(a, b):
                dict_with_replica.update(generate_replicas(num_replicas, file, ligand_name, mdtp))

        # Grompp Part
        struct_dict_run = premd_dict_gro
        top_dict = premd_dict_top
        mdp_files = self.mdp_files.value  # a List of Path or a Path
        valid_mdtps = [mdp.stem.capitalize() for mdp in mdp_files]
        self.logger.info(" Type of MD simulations will perform %s ", valid_mdtps)

        # EM, NVP, NPT or Prod?
        # First check if you get a list a mdp files or not
        # If its a list, then one MD followed by another
        if isinstance(mdp_files, list):
            mdp_files = [Path(path_str) for path_str in mdp_files]

            for mdfile in mdp_files:
                # Enumerate MD TYPE
                # All ligands with the same MD type will run mdrun in parallel
                md_commands = []
                lig_rep_tp_dirs = []
                mdtp = mdfile.stem.capitalize()

                for i, (k, v) in enumerate(struct_dict_run.items()):
                    # Each key:val pair in structure_dict_run represents one unique system
                    # defined by the number of replica and the name of ligand
                    # use this two as an index to access other files stored in one dict for this system
                    gro_run = v
                    replica_num = k[0]
                    ligand_name = k[1]
                    topology = top_dict[(replica_num, ligand_name)]
                    # Grompp needs itp files exit in the working directory
                    # You need to copy them into the grompp node
                    inp_protitp = premd_dict_protItp[(replica_num, ligand_name)]
                    inp_ligitp = premd_dict_ligItp[(replica_num, ligand_name)]

                    self.logger.info(
                        "Working on the %s-th system, ligand_name %s, replica_num %s,  mdtp %s",
                        i + 1,
                        ligand_name,
                        replica_num,
                        mdtp,
                    )

                    # Get the name of the itp files
                    name_top = topology.name
                    name_prot = inp_protitp.name
                    name_lig = inp_ligitp.name

                    # Node folder, current working directory
                    parent_workdir = self.work_dir
                    lig_rep_tp = parent_workdir / Path(ligand_name) / f"Replica{replica_num}" / mdtp
                    # Make subfolder
                    lig_rep_tp.mkdir(parents=True, exist_ok=True)
                    lig_rep_tp_dirs.append(lig_rep_tp)
                    self.logger.debug("1lig_rep_tp_dirs %s", lig_rep_tp_dirs)

                    # Check conditions
                    if self.set_posreProt.value:  # posre.itp
                        inp_posreProt = premd_dict_posreProt[(replica_num, ligand_name)]
                        if inp_posreProt.exists():
                            shutil.copy(inp_posreProt, lig_rep_tp / inp_posreProt.name)
                            posreProt_ori = Path(lig_rep_tp / inp_posreProt.name)
                            posreProt_md = extract_filename(
                                posreProt_ori, ligand_name, replica_num, valid_mdtps
                            )
                            if posreProt_md is not None:
                                shutil.move(posreProt_ori, lig_rep_tp / Path(posreProt_md))
                            else:
                                self.logger.error(
                                    "The associated file for key %s does not match any of the expected patterns",
                                    (replica_num, ligand_name),
                                )
                                continue

                    if self.set_posreLig.value:
                        inp_posreLig = premd_dict_posreLig[(replica_num, ligand_name)]
                        if inp_posreLig.exists():
                            shutil.copy(inp_posreLig, lig_rep_tp / inp_posreLig.name)
                            posreLig_ori = Path(lig_rep_tp / inp_posreLig.name)
                            posreLig = extract_filename(
                                posreLig_ori, ligand_name, replica_num, valid_mdtps
                            )
                            if posreLig is not None:
                                shutil.move(posreLig_ori, lig_rep_tp / Path(posreLig))
                            else:
                                self.logger.error(
                                    "The associated file for key %s does not match any of the expected patterns",
                                    (replica_num, ligand_name),
                                )
                                continue

                    if self.rst_file.value:
                        inp_rst = struct_dict_run[(replica_num, ligand_name)]

                    if self.index_file.value:
                        inp_index = premd_dict_index[(replica_num, ligand_name)]

                    # For each ligand, create new working dir
                    # Change name of topoly and itp files to standard name. e.g. topol.top
                    # as the itp file name in the topology file are hard coded
                    topology_copy = Path(lig_rep_tp / name_top)
                    shutil.copy(topology, topology_copy)
                    top0 = extract_filename(topology_copy, ligand_name, replica_num, valid_mdtps)

                    if top0 is not None:
                        top = lig_rep_tp / Path(top0)
                        shutil.move(topology_copy, top)
                    else:
                        self.logger.error(
                            "The associated file for key %s does not match any of the expected patterns",
                            (replica_num, ligand_name),
                        )
                        continue

                    # Manipulate protein and ligand .itp files , copy to the subfolder, extract name, move
                    protitp_copy = lig_rep_tp / name_prot  # name with replicas, mdtype
                    shutil.copy(inp_protitp, protitp_copy)
                    protitp = extract_filename(protitp_copy, ligand_name, replica_num, valid_mdtps)
                    if protitp is not None:
                        shutil.move(protitp_copy, lig_rep_tp / Path(protitp))
                    else:
                        self.logger.error(
                            "The associated file for key %s does not match any of the expected patterns",
                            (replica_num, ligand_name),
                        )
                        continue

                    ligitp_copy = lig_rep_tp / name_lig
                    shutil.copy(inp_ligitp, ligitp_copy)
                    ligitp = extract_filename(ligitp_copy, ligand_name, replica_num, valid_mdtps)
                    if ligitp is not None:
                        shutil.move(ligitp_copy, lig_rep_tp / Path(ligitp))
                    else:
                        self.logger.error(
                            "The associated file for key %s does not match any of the expected patterns",
                            (replica_num, ligand_name),
                        )
                        continue

                    # While running the job, all filenames convert to std fns
                    # Define output files
                    mdrun_file = lig_rep_tp / Path("topol.tpr")

                    # copy the mdp file to local dir so it can be processed and modified locally and simultanously
                    local_mdp = lig_rep_tp / mdfile.name
                    shutil.copy(mdfile, local_mdp)

                    mdp_file_parser = MDPFileParser(
                        local_mdp, replacements=self.new_mdp_values.value
                    )
                    mdp_file_parser.parse()

                    command_grompp2 = (
                        f"{self.runnable['gmx']} grompp "
                        f"-p {top.as_posix()} "
                        f"-c {gro_run.as_posix()} "
                        f"-f {local_mdp.as_posix()} "
                        f"-o {mdrun_file.as_posix()} "
                        f"-maxwarn {self.max_warn.value} "
                    )

                    if inp_rst.exists():
                        command_grompp2 += f"-r {inp_rst.as_posix()} "

                    if inp_index.exists():
                        command_grompp2 += f"-n {inp_index.as_posix()} "

                    # Run grompp
                    run_grompp2 = self.run_command(command_grompp2, working_dir=lig_rep_tp)
                    if run_grompp2.returncode != 0:
                        continue

                    # MD part
                    structure_out = lig_rep_tp / Path("confout.gro")
                    energy_out = lig_rep_tp / Path("ener.edr")
                    md_out = lig_rep_tp / Path("md.log")
                    cpt_prev_out = lig_rep_tp / Path("state_prev.cpt")
                    chk_out = lig_rep_tp / Path("state.cpt")
                    traj_out = lig_rep_tp / Path("traj.trr")
                    cmp_traj_out = lig_rep_tp / Path("traj.xtc")

                    num_steps = mdp_file_parser.num_steps
                    chk_steps = mdp_file_parser.chk_steps
                    nstxout_value = mdp_file_parser.last_nstxout_value
                    nstvout_value = mdp_file_parser.last_nstvout_value
                    nstfout_value = mdp_file_parser.last_nstfout_value
                    nstxtcout_value = mdp_file_parser.last_nstxtcout_value

                    md_command = (
                        f"{self.runnable['gmx']} mdrun "
                        f"-s {mdrun_file.as_posix()} "
                        f"-c {structure_out.as_posix()} "
                        f"-nt {self.num_threads.value} "
                        f"-ntomp {self.num_tomp.value} "
                        f"-ntmpi {self.num_tmpi.value} "
                        f"-e {energy_out.as_posix()} "
                        f"-g {md_out.as_posix()} "
                    )

                    if chk_steps < num_steps:
                        md_command += f"-cpo {chk_out.as_posix()} "
                        md_command += f"-cpt {self.cpt_interval.value} "

                    if nstxout_value or nstvout_value or nstfout_value != 0:
                        md_command += f"-o {traj_out.as_posix()} "

                    if nstxtcout_value != 0:
                        md_command += f"-x {cmp_traj_out.as_posix()} "

                    md_commands.append(md_command)

                # Run one type of MD for all ligands and replicas
                self.run_multi(
                    commands=md_commands,
                    working_dirs=lig_rep_tp_dirs,
                    verbose=True,
                    raise_on_failure=False,
                )
                dict_container = {}
                # Assume sendoutpotins are a list of std_fname
                for file_tp in [
                    mdrun_file,
                    top,
                    structure_out,
                    energy_out,
                    md_out,
                    traj_out,
                    cmp_traj_out,
                    chk_out,
                    cpt_prev_out,
                ]:
                    dict_file_tp = process_files(lig_rep_tp_dirs, mdtp, file_tp.name)

                    self.logger.debug("dict_file_tp %s", dict_file_tp)
                    dict_container[file_tp.stem + "_" + file_tp.suffix[1:]] = dict_file_tp
                    # topol_tpr
                    if file_tp.suffix[1:] == "gro":
                        struct_dict_run = dict_file_tp
                self.logger.debug("dict_container %s", dict_container)

            # Outputs
            sendout_options = self.sendout_option.value
            for option in sendout_options:
                data_dict = dict_container[option]
                getattr(self, f"out_{option}").send(data_dict)
                self.logger.info("Last MD run finished, sending out files in %s", data_dict)

            other_outputs = [
                "out_posreProt",
                "out_posreLig",
                "out_protItp",
                "out_ligItp",
                "out_index",
            ]
            fout_dicts = [
                premd_dict_posreProt,
                premd_dict_posreLig,
                premd_dict_protItp,
                premd_dict_ligItp,
                premd_dict_index,
            ]
            for out_option, data_dict in zip(other_outputs, fout_dicts):
                self.logger.info("data_dict %s", data_dict)
                convert_dict = convert_data_dict(data_dict)
                self.logger.debug("convert_dict %s", convert_dict)
                getattr(self, out_option).send(convert_dict)


@pytest.fixture
def protein_path(shared_datadir: Any) -> Any:
    return shared_datadir / "pdbs" / "1L83.pdb"


@pytest.fixture
def ligand_path(shared_datadir: Any) -> Any:
    return [shared_datadir / "ligs" / "ligand.pdb"]


@pytest.fixture
def mdp_path(shared_datadir: Any) -> Any:
    return shared_datadir / "mdps" / "em.mdp"


@pytest.fixture
def mdps_path(shared_datadir: Any) -> Any:
    return [shared_datadir / "mdps" / "em.mdp"]


class TestSuiteGmxMD:
    def test_MDs(
        self,
        temp_working_dir: Any,
        protein_path: Any,
        ligand_path: Any,
        mdp_path: Any,
        mdps_path: Any,
        test_config: Any,
    ) -> None:
        rig = TestRig(MDs, config=test_config)
        params: list[dict[str, Any]] = [
            {
                "ff": "amber03",
                "ff_wat": "tip3p",
                "replicas": 1,
                "charge_method": "gas",
                "box_type": "cubic",
                "mdp_file": mdp_path,
                "mdp_files": mdps_path,
                "cpt_interval": 1,
                "num_tomp": 16,
                "batch_options": JobResourceConfig(cores_per_process=1),
                "sendout_option": ["topol_tpr", "topol_top", "confout_gro", "ener_edr", "md_log"],
            }
        ]
        for param in params:
            res = rig.setup_run(
                inputs={"inp": [protein_path], "inp_lig": [ligand_path]}, parameters=param
            )
            tpr = res["out_topol_tpr"].get()
            top = res["out_topol_top"].get()
            confout = res["out_confout_gro"].get()
            log = res["out_md_log"].get()

            assert top is not None, "The 'top' variable is None."
            assert top.stat().st_size > 0.0

            assert confout is not None, "The 'confout' variable is None."
            assert confout.stat().st_size > 0.0

            assert tpr is not None, "The 'tpr' variable is None."
            assert tpr.stat().st_size > 0.0

            assert log is not None, "The 'log' variable is None."
            assert log.stat().st_size > 0.0
