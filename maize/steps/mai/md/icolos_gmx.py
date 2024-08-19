"""Legacy icolos support for standard GROAMCS MD """

# pylint: disable=import-outside-toplevel, import-error
from pathlib import Path
from typing import Annotated, ParamSpec, TypeVar

from maize.core.node import Node
from maize.core.interface import Input, FileParameter, Parameter, Suffix
from maize.utilities.chem import IsomerCollection, convert
from maize.utilities.io import sendtree
from maize.steps.mai.misc.icolos import _patch_icolos_conf as patch_icolos


_T = TypeVar("_T")
_P = ParamSpec("_P")


try:
    from maize.utilities.utilities import deprecated
except ImportError:
    from collections.abc import Callable
    from typing import Any

    def deprecated(_: str | None = None) -> Callable[[Any], Any]:  # type: ignore
        def inner(obj: Any) -> Any:
            return obj
        return inner


def pdbformat(atomline: str) -> str:
    """Converts a non-standard pdb line into standard format"""
    fields = atomline.split()
    formated = "%-6s%5d %4s %1s%6d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f      %4s%2s\n" % (
        fields[0],
        int(fields[1]),
        fields[2],
        fields[3],
        int(fields[4]),
        "",
        float(fields[5]),
        float(fields[6]),
        float(fields[7]),
        float(fields[8]),
        float(fields[9]),
        "",
        fields[10],
    )

    return formated


def file_fieldreplace(
    file_path: Path, search_str: str, replace_str: str, field_delimiter: str, displace: int
) -> None:
    """Replaces text in a file:
    Looks for matching string in file and exchanges the same field or a field next to it
    as defined by 'displace' with replacement string"""

    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(file_path, "w") as file:
        for line in lines:
            if search_str in line:
                fields = line.split(field_delimiter)
                for i, field in enumerate(fields):
                    if search_str in field:
                        fields[i + displace] = replace_str
                line = field_delimiter.join(fields)
            file.write(line)


@deprecated()
class IcolosGMX_MD(Node):
    """
    Running MD simulations with GROMACS through a series of simulation steps as
    defined in mdp simulation input files.
    Simulations are carried out through Icolos and each system is submitted
    as a seperate job through SLURM
    """

    # required_callables = ["icolos"]

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules to send to Icolos"""

    target: FileParameter[Annotated[Path, Suffix(".pdb")]] = FileParameter()
    """Target protein structure"""

    configuration: FileParameter[Annotated[Path, Suffix(".json")]] = FileParameter()
    """Icolos template file"""

    submission: FileParameter[Annotated[Path, Suffix(".sh")]] = FileParameter()
    """Template shell script to submit job to slurm"""

    mdps: FileParameter[Path] = FileParameter()
    """Folder containing mdp files for Gromacs"""

    replicas: Parameter[int] = Parameter(default=1)
    """Number of replicas to run for each system"""

    out: FileParameter[Path] = FileParameter()
    """Path to folder that contains all system subfolders with MD trajectories"""

    def run(self) -> None:
        mols = self.inp.receive()

        for mol in mols:
            for iso in mol.molecules:
                if iso is not None:
                    self.logger.info("Preparing mol %s for MD", iso)

                    # generate subfolders
                    moldir = Path(iso.inchi if iso.name is None else iso.name)
                    moldir.mkdir(parents=True, exist_ok=False)

                    moldir_out = moldir / "output"
                    moldir_out.mkdir(parents=True, exist_ok=True)

                    # convert ligand sdf into pdb format
                    iso.to_sdf(moldir / "mol.sdf")
                    convert(moldir / "mol.sdf", moldir / "mol.pdb", ["-m"])

                    # generate complex structure
                    with open(self.target.value, "r") as file:
                        complex_lines = []
                        for line in file:
                            if "ATOM" in line or "HETATM" in line:
                                complex_lines.append(line)

                    # fix ligand format
                    with open(moldir / "mol1.pdb", "r") as file:
                        for i, line in enumerate(file):
                            if "HETATM" in line:
                                pdb_fields = line.split()
                                pdb_fields[2] += str(i)
                                pdb_fields[3] = "MOL"
                                line = " ".join(pdb_fields)
                                line = pdbformat(line)
                                complex_lines.append(line)

                    complex_dir = moldir / "complex.pdb"
                    with open(complex_dir, "w") as file:
                        for line in complex_lines:
                            file.write(line)

                    # prepare json file
                    mdp_dir = self.mdps.value
                    global_vars = {
                        "file path": moldir.absolute().as_posix(),
                        "output_dir": moldir_out.absolute().as_posix(),
                        "complex": complex_dir.absolute().as_posix(),
                        "mdps": mdp_dir.absolute().as_posix(),
                    }
                    icoconf = moldir / "icolos_config.json"
                    patch_icolos(self.configuration.value, icoconf, global_vars)
                    file_fieldreplace(icoconf, "replicas", str(self.replicas.value), " ", 1)

                    # prepare submission script
                    sub_line = "icolos -debug -conf " + str(icoconf.absolute())
                    out = "#SBATCH -o " + str(moldir.absolute()) + "/out_%j.txt\n"
                    err = "#SBATCH -e " + str(moldir.absolute()) + "/err_%j.txt\n"

                    with open(self.submission.value, "r") as file:
                        submission_lines = []
                        for line in file:
                            if "out_" in line:
                                line = out
                            if "err_" in line:
                                line = err
                            submission_lines.append(line)
                        submission_lines.append(sub_line)

                    with open(moldir / "sub_script.sh", "w") as file:
                        for line in submission_lines:
                            file.write(line)

                    # submit job
                    command = "sbatch " + str(moldir / "sub_script.sh")
                    self.run_command(command, verbose=True)

                else:
                    self.logger.info("No isomer found in '%s'!", mol)

        sendtree({file.name: file for file in Path().iterdir()}, self.out.filepath)
