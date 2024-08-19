""""Schrodinger split_structure script interface"""

# pylint: disable=import-outside-toplevel, import-error

from abc import abstractmethod
from pathlib import Path
from typing import Annotated, List

from maize.core.interface import Input, Output, Parameter, Suffix
from maize.utilities.validation import SuccessValidator, FileValidator
from maize.steps.mai.common.schrodinger import Schrodinger


class SchrodingerSplitter(Schrodinger, register=False):
    """
    Calls Schrodinger's split_structure util to split an input file.
    Base class for more conveniently packaged nodes
    """

    required_callables = ["split_structure"]  # normally $SCHRODINGER/run split_structure.py

    inp: Input[Annotated[Path, Suffix("mae")]] = Input()
    """Protein to split, must be mae"""

    ligand_asl: Parameter[str] = Parameter(default="'res.ptype \"INH \"'")
    """ schrodinger ASL to select ligand, note the escaped quotes """

    split_waters: Parameter[bool] = Parameter(default=True)
    """ Remove waters from in receptor? setting to TRUE will remove waters"""

    mode: str
    """ type of split, must be set by derived classes"""

    def _process_extra_args(self) -> str:
        return ""

    @abstractmethod
    def _generate_generic_output_path(self, input_path: Path) -> Path:
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def _generate_target_output_paths(self, output_path: Path) -> List[Path]:
        raise NotImplementedError("To be implemented")

    @abstractmethod
    def handle_outputs(self, target_output_paths: List[Path]) -> None:
        raise NotImplementedError("To be implemented")

    def run(self) -> None:
        self.logger.info("starting splitter...")
        input_path = self.inp.receive()
        output_path = self._generate_generic_output_path(input_path)
        target_output_paths = self._generate_target_output_paths(output_path)
        validators = [SuccessValidator("Output split into files")] + [
            FileValidator(t) for t in target_output_paths
        ]
        self.logger.info(
            f"generating output {output_path},  fishing for"
            + " & ".join([t.as_posix() for t in target_output_paths])
            + f"with lig asl {self.ligand_asl.value}"
        )

        extra_commands = self._process_extra_args()
        extra_commands += "" if not self.split_waters.value else "-groupwaters "

        command = (
            f"{self.runnable['split_structure']}  -m {self.mode} -many_files "
            + extra_commands
            + f"-ligand_asl {self.ligand_asl.value} "
            + f"{input_path.as_posix()} {output_path.as_posix()}"
        )

        self.run_command(command, validators=validators, raise_on_failure=False, verbose=True)
        self.handle_outputs(target_output_paths)


class ProteinChainSplitter(SchrodingerSplitter):
    """
    Calls Schrodinger's split_structure util to extract a specific chain from a protein.
    """

    out: Output[Annotated[Path, Suffix("mae")]] = Output()
    """Path to generated protein file"""

    chain_to_keep: Parameter[str] = Parameter(default="A")
    """ chain to keep """

    mode = "chain"

    def _process_extra_args(self) -> str:
        return "-merge_ligands_with_chain "

    def _generate_generic_output_path(self, input_path: Path) -> Path:
        return input_path.with_stem(input_path.stem + "_chain_split")

    def _generate_target_output_paths(self, output_path: Path) -> List[Path]:
        return [output_path.with_stem(output_path.stem + "_chain" + self.chain_to_keep.value)]

    def handle_outputs(self, target_output_paths: List[Path]) -> None:
        self.out.send(target_output_paths[0])


class LigandProteinSplitter(SchrodingerSplitter):
    """
    Calls Schrodinger's split_structure util to split a ligand from protein.
    """

    required_callables = ["split_structure"]

    out_lig: Output[Annotated[Path, Suffix("mae")]] = Output()
    """Path to ligand file"""

    out_prot: Output[Annotated[Path, Suffix("mae")]] = Output()
    """Path to protein file"""

    mode = "ligand"

    def _generate_generic_output_path(self, input_path: Path) -> Path:
        return input_path.with_stem(input_path.stem + "_pl_split")

    def _generate_target_output_paths(self, output_path: Path) -> List[Path]:
        prot_output_path = output_path.with_stem(output_path.stem + "_receptor1")
        lig_output_path = output_path.with_stem(output_path.stem + "_ligand1")
        return [prot_output_path, lig_output_path]

    def handle_outputs(self, target_output_paths: List[Path]) -> None:
        self.out_prot.send(target_output_paths[0])
        self.out_lig.send(target_output_paths[1])
