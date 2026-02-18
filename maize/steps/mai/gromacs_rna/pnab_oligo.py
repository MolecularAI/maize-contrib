""" Node to generate 3D structures of chemically modified oligos """

from pathlib import Path
import shutil
import random
from collections import defaultdict
from typing import Any, cast
import pytest
from maize.core.node import Node
from maize.core.interface import Input, Output, FileParameter
from maize.utilities.io import Config
from maize.utilities.testing import TestRig
from maize.steps.mai.gromacs_rna.file_utils_genoligo import (
    replace_res,
    remove_p5,
    keep_p5,
    minimize,
)
from maize.steps.mai.gromacs_rna.common import constants


class GenOligo(Node):
    """
    Generates duplex structures of modified oligonucleotides.

    Notes
    -----
    - Requires a maize environment with ``pnab`` tool installed.
    The github repo for this tool can be found in `PNAB <https://github.com/GT-NucleicAcids/pnab>`_.
    To install use::
    conda install conda-forge::pnab

    - Requires openbabel. Github rep: 'openbabel <https://github.com/openbabel/openbabel>_.
    To install use::
    conda install conda-forge::openbabel

    -- Requires parmed. Link: <https://parmed.github.io/ParmEd/html/index.html>_.
    To install use::
    conda install conda-forge::parmed

    -- Requires pdb-tools. Link: <https://www.bonvinlab.org/pdb-tools/>_. To install use::
    pip install pdb-tools

    References
    -----
    .. Alenaizan, A., Barnett JL., Hud, NV., Sherrill, CD., Petrov, AS.
    "The proto-Nucleic Acid Builder: a software tool for constructing nucleic acid analogs"
    *Nucleic Acids Research*, 2021.
           URL: https://academic.oup.com/nar/article/49/1/79/6029180

    .. Rodrigues, J., Teixeira, JM., Trellet, M., Bonvin, AM.
    "pdb-tools: A swiss army knife for molecular structures"
    *F1000Res*, 2018.
           URL: https://pubmed.ncbi.nlm.nih.gov/30705752/

    """

    required_callables = ["pdb_reres"]

    required_packages = ["pnab", "openbabel", "parmed"]

    # Inputs

    inp_dict: Input[dict[str, list[str]]] = Input()
    """A list of strings defining 
    (a) base sequence, 
    (b) sugar modifications in strand1, 
    (c) backbone modifications in strand1, 
    (d) 5'terminal modification in strand1, 
    (e) sugar modifications in strand2, 
    (f) backbone modifications in strand2, 
    (g) 5'terminal modification in strand2. 
    
    Note: Make sure the same order is entries is maintained in the input file"""

    inp_data_path: Input[Path] = Input()
    """Path to the folder containing building blocks for construction (blocks)"""

    # Options

    inp_yaml: FileParameter[Path] = FileParameter()
    """Path to input config (PNAB) file"""

    # Outputs

    out_pdbs: Output[dict[str, Path]] = Output(mode="copy")
    """Generated pdb files as output"""

    def construct_seq(self, path: Path, bb: str, base: str, order: list[bool]) -> Any:
        """Construct sequence based on one type of modification"""
        import pnab

        yaml = self.inp_yaml.filepath
        if bb not in constants.feature_dict:
            self.logger.debug("Invalid modification, not found in blocks directory")
            return None
        bb_pdb = constants.feature_dict[bb]["bb_pdb"]
        bb_link = constants.feature_dict[bb]["bb_link"]
        bb_base_link = constants.feature_dict[bb]["bb_base_link"]

        mol = pnab.pNAB(f"{yaml}")
        mol.options["RuntimeParameters"]["strand"] = base
        mol.options["RuntimeParameters"]["build_strand"] = order
        mol.options["Backbone"]["file_path"] = f"{path}/blocks/{bb_pdb}"
        mol.options["Backbone"]["interconnects"] = bb_link
        mol.options["Backbone"]["linker"] = bb_base_link

        return mol

    def replacing(
        self,
        path: Path,
        cwd: Path,
        seq: list[str],
        seq_base: str,
        bb1: str,
        str_obmol: Any,
        order: list[bool],
        chain: str,
        strand: int,
    ) -> Any:
        """Replacing residues with modified residues"""
        from openbabel import openbabel as ob

        dict_mod = defaultdict(list)

        for pos, char in enumerate(seq[1:]):
            dict_mod[char].append(pos + 2)

        if bb1 in dict_mod:
            dict_mod.pop(bb1)

        conv = ob.OBConversion()

        junk_path = Path(f"{self.work_dir}/junk")
        fpatterns_to_move = ["*prefix.yaml", "*results.csv", "1_*.pdb"]

        if len(dict_mod) != 0:
            # Go over one modification after another in the modifications list
            count = 2
            for key in dict_mod:
                mod = self.construct_seq(path, key, seq_base, order)
                self.logger.info("Generating strand %s - modification %s", strand, count)
                mod.run()
                mod_obmol = ob.OBMol()
                conv.ReadFile(mod_obmol, f"{int(mod.results[0, 0])}_{int(mod.results[0, 1])}.pdb")
                final_mol = replace_res(key, dict_mod[key], str_obmol, mod_obmol, chain)
                conv.WriteFile(final_mol, f"{cwd}/temp.pdb")
                conv.ReadFile(str_obmol, f"{cwd}/temp.pdb")

                # Moving temporary files to junk
                for pattern in fpatterns_to_move:
                    files = self.work_dir.glob(pattern)
                    for file in files:
                        dst = junk_path / file.name
                        shutil.move(file, dst)

        # Naming terminal residues of strand
        for res_str in ob.OBResidueIter(str_obmol):
            residue_number = res_str.GetNum()
            if residue_number == 1:
                res_str.SetName(res_str.GetName().strip() + "5")
            if residue_number == str_obmol.NumResidues():
                res_str.SetName(res_str.GetName().strip() + "3")

        return str_obmol

    def run(self) -> None:
        import pnab
        from openbabel import openbabel as ob

        dict_rna = self.inp_dict.receive()
        data_path = self.inp_data_path.receive()
        output_dict = {}

        mol_count = 1
        for mol, value in dict_rna.items():

            seq_base = value[0]
            seq_sugar_guide = value[1]
            seq_bb_guide = value[2]
            guide_5mod = value[3]
            seq_sugar_pass = value[4]
            seq_bb_pass = value[5]
            pass_5mod = value[6]

            seq_guide = []
            seq_pass = []

            # Combining backbone and sugar modification strings
            # to pick structures of building blocks from blocks folder

            for index, item in enumerate(seq_bb_guide):
                seq_guide.append(item + seq_sugar_guide[index])

            for index, item in enumerate(seq_bb_pass):
                seq_pass.append(item + seq_sugar_pass[index])

            self.logger.info("MAKING MOLECULE %s...", mol_count)

            # strand 1 - guide
            bb1_guide = seq_guide[0]
            order_guide = [True, False, False, False, False, False]
            str1 = self.construct_seq(
                data_path, bb1_guide, seq_base, order_guide
            )  # Construct sequence based on the first modification along the sequence

            # strand 2 - passenger / target
            bb1_pass = seq_pass[0]
            order_pass = [False, True, False, False, False, False]
            str2 = self.construct_seq(
                data_path, bb1_pass, seq_base, order_pass
            )  # Construct sequence based on the first modification along the sequence

            conv = ob.OBConversion()
            str1_obmol = ob.OBMol()
            str2_obmol = ob.OBMol()

            # Run strand1 and read the result
            self.logger.info("Generating strand1 - modification 1")
            str1.run()
            conv.ReadFile(str1_obmol, f"{int(str1.results[0, 0])}_{int(str1.results[0, 1])}.pdb")

            # Run strand2 and read the result
            self.logger.info("Generating strand2 - modification 1")
            str2.run()
            conv.ReadFile(str2_obmol, f"{int(str2.results[0, 0])}_{int(str2.results[0, 1])}.pdb")

            # Codes from the pnab input file translated to residue names in gromacs topology.
            # Names are in correspondence to siRNA database
            dict_resnames = {
                "pr": " ",
                "sr": "P",
                "rr": "P",
                "pm": "M",
                "sm": "B",
                "pf": "F",
                "sf": "H",
                "pg": "G",
                "pd": "D",
                "sd": "S",
                "rd": "S",
                "pe": "E",
                "se": "O",
                "pl": "L",
                "sl": "N",
                "rl": "N",
            }

            # Setting residue names for strand1
            for index, res_str1 in enumerate(ob.OBResidueIter(str1_obmol)):
                res_str1.SetName(dict_resnames[bb1_guide] + res_str1.GetName().strip())
                current_atom = str1_obmol.GetAtom(str1_obmol.NumAtoms())
                current_atom.SetResidue(res_str1)

            # Setting residue names for strand2
            for index, res_str2 in enumerate(ob.OBResidueIter(str2_obmol)):
                res_str2.SetName(dict_resnames[bb1_pass] + res_str2.GetName().strip())
                current_atom = str2_obmol.GetAtom(str2_obmol.NumAtoms())
                current_atom.SetResidue(res_str2)

            # Moving temporary files to junk folder
            Path(f"{self.work_dir}/junk").mkdir(parents=True, exist_ok=True)
            junk_path = Path(f"{self.work_dir}/junk")
            fpatterns_to_move = ["*prefix.yaml", "*results.csv", "1_*.pdb"]

            for pattern in fpatterns_to_move:
                files = self.work_dir.glob(pattern)
                for file in files:
                    dst = junk_path / file.name
                    shutil.move(file, dst)

            # Generating strand1 - guide strand
            chain_str1 = "A"
            mol_str1 = self.replacing(
                data_path,
                self.work_dir,
                seq_guide,
                seq_base,
                bb1_guide,
                str1_obmol,
                order_guide,
                chain_str1,
                1,
            )
            # Keeping or removing phosphate in the 5'terminal end
            if guide_5mod == "p":
                mol_str1 = keep_p5(self.work_dir, mol_str1)
            elif guide_5mod == "n":
                mol_str1 = remove_p5(self.work_dir, mol_str1)

            # Generating strand2 - passenger/target strand
            chain_str2 = "B"
            mol_str2 = self.replacing(
                data_path,
                self.work_dir,
                seq_pass,
                seq_base,
                bb1_pass,
                str2_obmol,
                order_pass,
                chain_str2,
                2,
            )
            # Keeping or removing phosphate in the 5'terminal end
            if pass_5mod == "p":
                mol_str2 = keep_p5(self.work_dir, mol_str2)
            elif pass_5mod == "n":
                mol_str2 = remove_p5(self.work_dir, mol_str2)

            final_mol = mol_str1

            # Add the two strands
            final_mol += mol_str2
            final_mol.SetChainsPerceived()
            # Write and read the information.
            # This allows openbabel to get correct OBResidue information and number
            conv.WriteFile(final_mol, f"{self.work_dir}/temp.pdb")
            conv.ReadFile(final_mol, f"{self.work_dir}/temp.pdb")
            num_residues_chain = final_mol.NumResidues() // 2

            # Energy minimization of the structure

            self.logger.info("Optimizing structure...")
            duplex = minimize(final_mol, num_residues_chain)
            self.logger.info("Structure generated!")

            conv.WriteFile(duplex, f"{self.work_dir}/temp.pdb")

            # Renumbering residues

            proc = f"{self.runnable['pdb_reres']} {self.work_dir}/temp.pdb"
            pdb_out = self.run_command(proc).stdout
            with open(f"{self.work_dir}/{mol}.pdb", "wb") as pdbfile_out:
                pdbfile_out.write(pdb_out)

            # Moving temporary files to junk
            shutil.move(f"{self.work_dir}/temp.pdb", f"{junk_path}/temp.pdb")

            output_dict[mol] = Path(f"{mol}.pdb")

            mol_count += 1

        # Sending output pdbs to saving node
        self.out_pdbs.send(output_dict)


@pytest.fixture
def data_path(shared_datadir: Path) -> list[Path]:
    return [shared_datadir]


@pytest.fixture
def seq_data(shared_datadir: Path) -> Path:
    return shared_datadir / "sequence_data.csv"


@pytest.fixture
def yaml_path(shared_datadir: Path) -> Path:
    return shared_datadir / "input.yaml"


class TestSuitePNAB:
    def test_PNAB(
        self,
        data_path: Path,
        seq_data: Path,
        yaml_path: Path,
        test_config: Config,
    ) -> None:
        import pandas as pd

        rig = TestRig(GenOligo, config=test_config)
        df_seq = pd.read_csv(f"{seq_data}")
        random_num = random.randint(1, len(df_seq))

        df_dict: dict[str, list[str]] = {}

        for index, row in df_seq.iterrows():
            index = cast(int, index)
            df_dict[f"mol{index+1}"] = list(row)

        res = rig.setup_run(
            inputs={"inp_data_path": data_path, "inp_dict": [df_dict]},
            parameters={"inp_yaml": yaml_path},
        )

        pdb = res["out_pdbs"].get()

        assert pdb is not None
        assert pdb[f"mol{random_num}"].stat().st_size > 0.0
