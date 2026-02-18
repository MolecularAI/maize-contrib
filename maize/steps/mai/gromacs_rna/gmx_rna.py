"""MD simulations for oligos using Gromacs"""

from pathlib import Path
import shutil
from typing import Any, Literal
import pytest
from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, FileParameter, Flag
from maize.utilities.testing import TestRig
from maize.utilities.io import Config
from maize.steps.mai.gromacs.file_utils import (
    MDPFileParser,
    generate_replicas,
    get_index,
    process_files,
)
from maize.utilities.execution import JobResourceConfig


class MDsRNA(Node):
    """
    MD simulations conducted by GROMACS.
    It includes pdb2gmx, editconf, solvate, genion, make_ndx, grompp and mdrun commands.
    The force-field is taken from a custom force-field (based on the Amber14SB_OL15 FF)
    created for modified RNA and DNA residues.
    Notes
    -----
    This Node includes essential commands to start a gmx MD run.
    For questions about GROMACS, please visit its official documentation at
    https://manual.gromacs.org/current/index.html

    """

    required_callables = ["gmx"]

    # Inputs

    inp_rna: Input[list[Path]] = Input()
    """A list of file paths of RNA structure files. Accept files in PDB formats."""

    ff: FileParameter[Path] = FileParameter()
    """"pdb2gmx: Force field for modified RNA. 
    Path to Custom-made force-field."""

    ff_wat: Parameter[str] = Parameter(default="tip3p")
    """"pdb2gmx: Force field for solvent. """

    # Outputs

    out_topol_tpr: Output[dict[tuple[int, str], Path]] = Output(mode="copy")
    """Tpr file as output"""

    out_topol_top: Output[dict[str, Path]] = Output(mode="copy")
    """Top file as output"""

    out_confout_gro: Output[dict[tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Structure file"""

    out_ener_edr: Output[dict[tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Structure file"""

    out_state_cpt: Output[dict[tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Checkpoint file"""

    out_state_prev_cpt: Output[dict[tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Previous state checkpoint file"""

    out_traj_trr: Output[dict[tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Trajectory file"""

    out_traj_xtc: Output[dict[tuple[int, str], Path]] = Output(mode="copy", optional=True)
    """Compressed trajectory file"""

    out_md_log: Output[dict[tuple[int, str], Path]] = Output(mode="copy")
    """MD log file"""

    out_ch_posre: Output[dict[str, list[Path]]] = Output(mode="copy", optional=True)
    """position rst file"""

    out_ch_itp: Output[dict[str, list[Path]]] = Output(mode="copy")
    """topology .itp files for individual chains of the molecules"""

    out_index_ndx: Output[dict[str, Path]] = Output(mode="copy", optional=True)
    """Index file"""

    # Options

    rst_file: Flag = Flag(default=True)
    """mdrun: Structure file: gro"""

    index_file: Flag = Flag(default=True)
    """mdrun: Index file"""

    replicas: Parameter[int] = Parameter(default=1)
    """generate_replicas: Number of replicas"""

    ignore_Hatoms: Flag = Flag(default=False)
    """pdb2gmx: Ignore hydrogen atoms that are in the coordinate file.
    If you wish to maintain the protonation state, you should use "False"."""

    box_type: Parameter[Literal["triclinic", "cubic", "octahedron", "dodecahedron"]] = Parameter(
        default="cubic"
    )
    """editconf: Box type"""

    distance: Parameter[float] = Parameter(default=1.2)
    """editconf: Distance between the solute and the box"""

    solvent_model: Parameter[str] = Parameter(default="spc216")
    """solvate: Structure file for solvent. 
    Options can be found in path/to/gromcas/top folder"""

    mdp_file: FileParameter[Path] = FileParameter()
    """grompp: Grompp input file with MD parameters"""

    max_warn: Parameter[int] = Parameter(default=0)
    """grompp: Number of allowed warnings during input processing."""

    replace_with: Parameter[str] = Parameter(default="SOL")
    """genion: Replace solvent molecules with monoatomic ions"""

    ion_conc: Parameter[float] = Parameter(default=0)
    """genion: Specify salt concentration of solution (in mol/litre)"""

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

    cpt_interval: Parameter[int] = Parameter(default=15)
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

    def run(self) -> None:

        rnas = self.inp_rna.receive()
        num_replicas = self.replicas.value
        ff_in = self.ff.filepath
        wat_in = self.ff_wat.value
        parent_workdir = self.work_dir
        self.logger.info("Oligo structures received %s", rnas)
        self.logger.info("Number of replicas will be excuted on each oligo %s", num_replicas)

        # Force-field needs to be in the cwd
        shutil.copytree(ff_in, parent_workdir / ff_in.name, dirs_exist_ok=True)
        shutil.copy(ff_in.parent / "residuetypes.dat", parent_workdir)

        premd_dict_posre: dict[str, Any] = {}
        premd_dict_itp: dict[str, Any] = {}
        premd_dict_index: dict[str, Any] = {}
        premd_dict_top: dict[str, Any] = {}
        premd_dict_gro: dict[tuple[int, str], Any] = {}

        for rna in rnas:
            rna_name = rna.stem
            self.logger.info("Starting with %s", rna)
            # Create subfolder name with RNA_name
            rna_sub = Path(rna_name).absolute()

            self.logger.debug(
                "Making a sub-folder for oligo %s at %s", rna_name, rna_sub.as_posix()
            )
            rna_sub.mkdir(parents=True, exist_ok=True)
            if not rna_sub.exists():
                self.logger.debug("Failed to create the subfolder")
                continue
            # Force-field needs to be in the cwd
            shutil.copytree(ff_in, rna_sub / ff_in.name, dirs_exist_ok=True)
            shutil.copy(ff_in.parent / "residuetypes.dat", rna_sub)

            structure_out = rna_sub / f"{rna_name}.gro"
            topology_out = rna_sub / "topol.top"
            posre_out = rna_sub / "posre.itp"

            if not self.ignore_Hatoms.value:
                self.logger.debug(" pdb2gmx self.workdir() %s ", parent_workdir)
                self.logger.debug(" pdb2gmx structure_in.absolute().exists() %s ", rna.exists())
                command_pdb2gmx = (
                    f"{self.runnable['gmx']} pdb2gmx "
                    f"-f {rna.as_posix()} "
                    f"-water {wat_in.strip() } "
                    f"-ff {ff_in.name[:-3] } "
                    f"-o {structure_out.as_posix()} "
                    f"-p {topology_out.as_posix()} "
                    f"-i {posre_out.as_posix()} "
                )
            else:
                command_pdb2gmx = (
                    f"{self.runnable['gmx']} pdb2gmx "
                    f"-f {rna.as_posix()} "
                    f"-water {wat_in.strip() } "
                    f"-ff {ff_in.name[:-3] }"
                    f"-o {structure_out.as_posix()} "
                    f"-p {topology_out.as_posix()} "
                    f"-i {posre_out.as_posix()} "
                    f"-ignh "
                )
            self.run_command(command_pdb2gmx, working_dir=rna_sub)
            self.logger.debug(
                "RNA.pdb is generated %s and saved in %s",
                structure_out.exists(),
                structure_out.absolute(),
            )

            itp_ch_out = [file.absolute() for file in rna_sub.glob("topol*.itp")]
            posre_ch_out = [file.absolute() for file in rna_sub.glob("posre*.itp")]

            # Editconf
            editconf_in = structure_out
            editconf_out = rna_sub / "box.gro"
            command_editconf = (
                f"{self.runnable['gmx']} editconf "
                f"-f {editconf_in.as_posix()} "
                f"-o {editconf_out.as_posix()} "
                f"-bt {self.box_type.value } "
                f"-d {self.distance.value } "
                f"-c "
            )

            run_editconf = self.run_command(command_editconf, raise_on_failure=False)
            if run_editconf.returncode != 0:
                continue

            # Solvate
            solvate_gro_in = editconf_out
            solvate_io = rna_sub / "solvated.gro"
            topology_io = topology_out
            command_solvate = (
                f"{self.runnable['gmx']} solvate "
                f"-cp {solvate_gro_in.as_posix()} "
                f"-cs {self.solvent_model.value} "
                f"-o {solvate_io.as_posix()} "
                f"-p {topology_io.as_posix()} "
            )
            run_solvate = self.run_command(command_solvate, raise_on_failure=False)

            if run_solvate.returncode != 0:
                continue

            # Grompp
            grompp_gro_in = solvate_io
            grompp_out_mdrun = rna_sub / "topol.tpr"
            from_mdp = self.mdp_file.filepath
            mdp_local = from_mdp.name
            shutil.copy(from_mdp, mdp_local)
            grompp_command = (
                f"{self.runnable['gmx']} grompp "
                f"-p {topology_io.as_posix()} "
                f"-c {grompp_gro_in.as_posix()} "
                f"-f {from_mdp.as_posix()} "
                f"-maxwarn {self.max_warn.value} "
                f"-o {grompp_out_mdrun.as_posix()} "
            )
            run_grompp = self.run_command(grompp_command, raise_on_failure=False)
            if run_grompp.returncode != 0:
                continue

            # MakeIndex
            mkndx_gro_in = grompp_gro_in
            ndx = rna_sub / "index.ndx"
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
            genion_gro = rna_sub / "confout.gro"
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
                f"-conc {self.ion_conc.value} "
                f"-o {genion_gro.as_posix()} "
                f"-p {genion_top.as_posix()} "
                f"-s {genion_tpr.as_posix()} "
                f"-n {ndx.as_posix()} "
            )
            run_genion = self.run_command(
                command_genion,
                command_input=f"{idx_sol}" + "\n" + "q" + "\n",
                verbose=True,
                raise_on_failure=False,
            )
            if run_genion.returncode != 0:
                continue

            # MakeIndex
            mkndx_gro_in = genion_gro
            ndx = rna_sub / "index.ndx"
            command_mkndx = (
                f"{self.runnable['gmx']} make_ndx "
                f"-f {mkndx_gro_in.as_posix()} "
                f"-o {ndx.as_posix()} "
            )
            run_ndx = self.run_command(
                command_mkndx, command_input="q\n", verbose=True, raise_on_failure=False
            )
            if run_ndx.returncode != 0:
                continue
            self.logger.info("Make_ndx wrote index file at %s", ndx.absolute())

            mdtp_ion = self.mdp_files.value
            mdtp = mdtp_ion[0].stem.capitalize()

            # Generating dictionaries with paths to files
            a = [premd_dict_gro]
            b = [genion_gro.absolute()]

            for dict_with_replica, file_rep in zip(a, b):
                dict_with_replica.update(generate_replicas(num_replicas, file_rep, rna_name, mdtp))

            c = [premd_dict_top, premd_dict_itp, premd_dict_posre, premd_dict_index]
            d = [genion_top.absolute(), itp_ch_out, posre_ch_out, ndx.absolute()]
            for dict_with_mol, file_mol in zip(c, d):
                if rna_name in dict_with_mol:
                    dict_with_mol[rna_name].extend(file_mol)
                else:
                    dict_with_mol.update({rna_name: file_mol})

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
                # All oligos with the same MD type will run mdrun in parallel
                md_commands = []
                rna_rep_tp_dirs = []
                mdtp = mdfile.stem.capitalize()

                for i, (k, v) in enumerate(struct_dict_run.items()):
                    # Each key:val pair in structure_dict_run represents one unique system
                    # defined by the number of replica and the name of oligo
                    # use this two as an index to access other files
                    # stored in one dict for this system
                    gro_run = v
                    replica_num = k[0]
                    rna_name = k[1]
                    topology = top_dict[rna_name]

                    self.logger.info(
                        "Working on the %s-th system, oligo_name %s, replica_num %s,  mdtp %s",
                        i + 1,
                        rna_name,
                        replica_num,
                        mdtp,
                    )

                    # Node folder, current working directory
                    parent_workdir = self.work_dir
                    rna_rep_tp = Path(rna_name) / f"Replica{replica_num}" / mdtp
                    # Make subfolder
                    rna_rep_tp.mkdir(parents=True, exist_ok=True)
                    rna_rep_tp_dirs.append(rna_rep_tp)
                    self.logger.debug("rna_rep_tp_dirs %s", rna_rep_tp_dirs)

                    if self.rst_file.value:
                        inp_rst = struct_dict_run[(replica_num, rna_name)]

                    if self.index_file.value:
                        inp_index = premd_dict_index[rna_name]

                    # Define output files
                    mdrun_file = rna_rep_tp / Path("topol.tpr")

                    # copy the mdp file to local dir so it can be
                    # processed and modified locally and simultanously
                    local_mdp = rna_rep_tp / mdfile.name
                    shutil.copy(mdfile, local_mdp)

                    mdp_file_parser = MDPFileParser(
                        local_mdp, replacements=self.new_mdp_values.value
                    )
                    mdp_file_parser.parse()

                    command_grompp2 = (
                        f"{self.runnable['gmx']} grompp "
                        f"-p {topology.as_posix()} "
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
                    run_grompp2 = self.run_command(command_grompp2, verbose=True)
                    if run_grompp2.returncode != 0:
                        continue

                    # MD part
                    structure_out = rna_rep_tp / Path("confout.gro")
                    energy_out = rna_rep_tp / Path("ener.edr")
                    md_out = rna_rep_tp / Path("md.log")
                    cpt_prev_out = rna_rep_tp / Path("state_prev.cpt")
                    chk_out = rna_rep_tp / Path("state.cpt")
                    traj_out = rna_rep_tp / Path("traj.trr")
                    cmp_traj_out = rna_rep_tp / Path("traj.xtc")

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
                        f"-o {traj_out.as_posix()} "
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

                # Run one type of MD for all oligos and replicas
                self.run_multi(
                    commands=md_commands,
                    verbose=True,
                    raise_on_failure=False,
                )

                dict_container: dict[str, dict[tuple[int, str], Path]] = {}
                dict_container_other: dict[str, dict[str, Path]] = {}

                # Assume sendoutpotins are a list of std_fname
                for file_tp in [
                    mdrun_file,
                    structure_out,
                    energy_out,
                    md_out,
                    traj_out,
                    cmp_traj_out,
                    chk_out,
                    cpt_prev_out,
                ]:
                    dict_file_tp = process_files(rna_rep_tp_dirs, mdtp, file_tp.name)

                    self.logger.debug("dict_file_tp %s", dict_file_tp)
                    dict_container[file_tp.stem + "_" + file_tp.suffix[1:]] = dict_file_tp

                    if file_tp.suffix[1:] == "gro":
                        struct_dict_run = dict_file_tp
                self.logger.debug("dict_container %s", dict_container)

            # Outputs
            sendout_options = self.sendout_option.value

            other_outputs = ["topol_top", "ch_itp", "ch_posre", "index_ndx"]
            fout_dicts = [premd_dict_top, premd_dict_itp, premd_dict_posre, premd_dict_index]

            for out_option, data_dict_fout in zip(other_outputs, fout_dicts):
                dict_container_other[out_option] = data_dict_fout
                self.logger.debug("dict_container_other %s", dict_container_other)

            for option in sendout_options:
                if option in dict_container:
                    data_dict = dict_container[option]
                    getattr(self, f"out_{option}").send(data_dict)
                    self.logger.info("Last MD run finished, sending out files in %s", data_dict)

                if option in dict_container_other:
                    data_dict_other = dict_container_other[option]
                    getattr(self, f"out_{option}").send(data_dict_other)
                    self.logger.info(
                        "Last MD run finished, sending out files in %s", data_dict_other
                    )


@pytest.fixture
def rna_path(shared_datadir: Path) -> list[Path]:
    return [shared_datadir / "rna_ss_pdbs" / "mol1.pdb"]


@pytest.fixture
def mdp_path(shared_datadir: Path) -> Path:
    return shared_datadir / "mdps" / "ions.mdp"


@pytest.fixture
def mdps_path(shared_datadir: Path) -> list[Path]:
    return [shared_datadir / "mdps" / "em.mdp"]


class TestSuiteGmxMDRNA:
    def test_MDsRNA(
        self,
        rna_path: list[Path],
        mdp_path: Path,
        mdps_path: list[Path],
        test_config: Config,
    ) -> None:
        rig = TestRig(MDsRNA, config=test_config)
        params: list[dict[str, Any]] = [
            {
                "ff_wat": "tip3p",
                "replicas": 1,
                "box_type": "cubic",
                "mdp_file": mdp_path,
                "mdp_files": mdps_path,
                "num_tomp": 4,
                "batch_options": JobResourceConfig(cores_per_process=1),
                "sendout_option": ["topol_top", "topol_tpr", "confout_gro", "ener_edr", "md_log"],
            }
        ]

        for param in params:
            res = rig.setup_run(inputs={"inp_rna": [rna_path]}, parameters=param)

            top = res["out_topol_top"].get()

            tpr = res["out_topol_tpr"].get()

            gro = res["out_confout_gro"].get()

            edr = res["out_ener_edr"].get()

            log = res["out_md_log"].get()

            assert top is not None
            assert top[f"{rna_path[0].stem}"].stat().st_size > 0.0

            assert tpr is not None
            assert tpr[(params[0]["replicas"], f"{rna_path[0].stem}")].stat().st_size > 0.0

            assert gro is not None
            assert gro[(params[0]["replicas"], f"{rna_path[0].stem}")].stat().st_size > 0.0

            assert edr is not None
            assert edr[(params[0]["replicas"], f"{rna_path[0].stem}")].stat().st_size > 0.0

            assert log is not None
            assert log[(params[0]["replicas"], f"{rna_path[0].stem}")].stat().st_size > 0.0
