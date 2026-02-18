"""
This file contains the node ReactionControl in Maize and some helper function and classes to
correctly running it. This node takes a reaction library csv file as input and 
automatically creates geometries for the molecular structures involved in the reaction. 
It is based on mechanistic templates stored as sdf files. After generating the structure 
the node sends the list of molecular structures for calculations. 
The type of reaction has to be specified by the user through the 'reaction' parameter. List of available 
reactions in template_repo.py.

The ReactionControl node receives inputs in the form of a list of molecular structures 
(List of Isomer Collections), from the calculations nodes. 
Function to evaluate the status of the calculations are present, failed calculations are
resubmitted. 

Successful calculations are then stored and information is used to build potential energy surfaces of the
reaction studied.

"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Any, cast, Literal
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from rdkit import Chem
from rdkit.Chem import AllChem, FragmentMatcher

from pathlib import Path
from maize.core.node import Node
from maize.core.interface import Input, Output, Parameter, FileParameter
from maize.utilities.testing import TestRig
from maize.utilities.chem.chem import Isomer, IsomerCollection, Conformer
import maize.steps.mai.molecule.template_repo as template_repo
from maize.steps.mai.molecule.compchem_utils import (
    Loader,
    AtomEntry,
    ConfTag,
    check_connectivity,
    check_refined,
    check_collaps
)

log = logging.getLogger("run")

def split_smiles_from_reaction(smiles: str) -> list[str]:
    """
    Split a part of reaction SMILES, e.g. reactants or products
    into components. Taking care of intra-molecular complexes

    Taken from RDKit:
    https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/ChemReactions/DaylightParser.cpp

    :param smiles: the SMILES/SMARTS
    :return: the individual components.
    """
    pos = 0
    block_start = 0
    level = 0
    in_block = 0
    components = []
    while pos < len(smiles):
        if smiles[pos] == "(":
            if pos == block_start:
                in_block = 1
            level += 1
        elif smiles[pos] == ")":
            if level == 1 and in_block:
                in_block = 2
            level -= 1
        elif level == 0 and smiles[pos] == ".":
            if in_block == 2:
                components.append(smiles[block_start + 1 : pos - 1])
            else:
                components.append(smiles[block_start:pos])
            block_start = pos + 1
            in_block = 0
        pos += 1
    if block_start < pos:
        if in_block == 2:
            components.append(smiles[block_start + 1 : pos - 1])
        else:
            components.append(smiles[block_start:pos])
    return components


class ReactionControl(Node):
    """
    Takes a reaction library (CSV) row as input. Creates an instance with all
    the relevant molecule objects for the reaction.  It outputs the molecular
    structures for each species and the respective configuration files for the
    following calculations.

    """
    tags = {"chemistry", "reaction"}

    inp: Input[pd.Series] = Input()
    """ Input from experimental record as pandas Series"""

    out: Output[list[IsomerCollection]] = Output()
    """ Final output. JSON file with final calculations """

    out_crest: Output[list[IsomerCollection]] = Output()
    """ Connected to Crest node input, sends molecules for which to generate conformers """

    inp_crest: Input[list[IsomerCollection]] = Input()
    """ Connected to Crest node output, receives the generated conformers """

    out_xg16: Output[list[IsomerCollection]] = Output()
    """ Connected to xTB and Gaussian nodes input (chained), sends molecules to calculate the gibbs free energy for """

    inp_xg16: Input[list[IsomerCollection]] = Input()

    """ Input from the calculation node with calculations results to be assessed """

    reaction: Parameter[Literal["NiCatCycle_CN", "NiCatCycle_CC", "NiCatCycle_CC2",
                                "RadicalTransformations",
                                "ConditionalRadicalTransformations",
                                "Oxidize",
                                "OxidizeandReduce"]] = Parameter(default="NiCatCycle_CC2")
    """Reaction template to use"""

    requested_components: Parameter[list[str]] = Parameter(default=['all'])

    _cache: dict[str, float]

    def get_generics(self, row: pd.Series) -> None:
        """
        Stores some informations about the reaction present
        in the csv file as well as the reaction role string.

        """
        
        self.id = row["Libname"]  # id of the reaction library
        self.rsmi = row["rsmi_processed"]  # reaction smiles
        if (self.reaction.value == 'NiCatCycle_CC') or (self.reaction.value == 'NiCatCycle_CC2'):
            self.others = row["CorrectedReagentsSmiles"]

    @staticmethod
    def assess_crest_calculations(calc: IsomerCollection) -> tuple[bool, str]:
        """
        Checks the status of a Crest calculations and
        returns True if not finished/converged correctly.

        Removes conformers from conformers list that don't match the
        parent molecule connectivity.

        Parameters
        ----------
        calc
            The output of the calculation node

        Returns
        -------
        bool
            Boolean with status of Crest calculation
        message
            string with information about the calculation status
        """

        crest_status = True
        message = "CREST calculation failed."
        for i in range(len(calc.molecules)):
            isomer_rd_mol = calc.molecules[i]._molecule
            isomer_structure = Loader.molecule_from_rdkit(isomer_rd_mol)

            if calc.molecules[i].get_tag("crest_exit_code") == 0:
                message = "CREST finished correctly."
                n_conf = len(calc.molecules[i].conformers)
                for indx, conf in enumerate(calc.molecules[i].conformers):
                    conf_structure = Loader.molecule_from_conformer(conf)

                    if not check_connectivity(isomer_structure, conf_structure):
                        refined, clashes = check_refined(isomer_structure, conf_structure)
                        if not refined:
                            message += " Clashes from raw geometry not removed."
                            calc.molecules[i].remove_conformer(indx)
                        elif refined and clashes:
                            message += " Refined geometries but still with clashes, another optimisation needed."
                        else:
                            message += " Connectivity is changed because of optimised clashes."


                if len(calc.molecules[i].conformers) > 0:
                    n_conf_removed = n_conf - len(calc.molecules[i].conformers)
                    crest_status = False
                    message += f" {n_conf_removed} conformer(s) were removed from {calc.molecules[0].get_tag('component')} for connectivity errors."
                else:
                    message += f" However, all conformers were removed from {calc.molecules[0].get_tag('component')} due to connectivity errors."
        return crest_status, message

    @staticmethod
    def assess_xtb_calculations(calc: IsomerCollection) -> tuple[bool, str]:
        """
        Checks the status of a XTB calculations and
        returns True if not finished/converged correctly.

        Removes conformers from tag list that don't match the
        parent molecule connectivity.

        Parameters
        ---------------------------
        calc
            The output of the calculation node

        Returns
        -------
        bool
            Boolean with status of xtb calculation
        """

        xtb_status = True
        
        message = "XTB calculation failed."
        
        for i in range(len(calc.molecules)):
            isomer_obj = calc.molecules[i]              
            isomer_exit_codes = json.loads(cast(str, isomer_obj.get_tag("XTB_exit_codes")))

            if 0 in isomer_exit_codes.values():
                message = "XTB finished correctly."
                isomer_rd_mol = isomer_obj._molecule
                isomer_structure = Loader.molecule_from_rdkit(isomer_rd_mol)
                
                conf_geometries_tag = json.loads(cast(str, isomer_obj.get_tag("XTB_geometries")))
                n_conf = len(conf_geometries_tag)
                message += f" There are this many conformations: {n_conf}. "
                connectivity_string = f"{isomer_obj.get_tag('component')}\n"
                for idx in range(len(conf_geometries_tag)):
                    xtb_conf_json = conf_geometries_tag[str(idx)]
                    if isinstance(xtb_conf_json, str):
                        message += f" {isomer_exit_codes.values()}. XTB calc failed for conformer {idx}."
                        continue
                    else:
                        xtb_conf_structure = Loader.molecule_from_json(xtb_conf_json, f"conformer-{idx}")

                        parent_conf = calc.molecules[i].conformers[idx]
                        parent_conf_structure = Loader.molecule_from_conformer(parent_conf)
                        
                        if isomer_obj.has_tag('ar_carbon'):
                            
                            if check_collaps(xtb_conf_structure, cast(int, isomer_obj.get_tag('ar_carbon')), cast(int, isomer_obj.get_tag('alk_carbon'))):
                                
                                message += f"conformer-{idx} of isomer {isomer_obj.get_tag('component')} collapsed to product"
                                del conf_geometries_tag[str(idx)]
                            

                        if not check_connectivity(parent_conf_structure, xtb_conf_structure):
                            message += f"Warning conformer-{idx} has changed connectivity compared to parent molecule. Please double check."                                


                if len(conf_geometries_tag) > 0:
                    n_conf_removed = n_conf - len(conf_geometries_tag)
                    xtb_status = False
                    message += f" {isomer_obj.get_tag('component')} {n_conf_removed} conformer(s) were removed for connectivity errors."
               
                else:
                    message += f" However, all conformers for {isomer_obj.get_tag('component')} were removed due to connectivity errors. See connectivity:\n{connectivity_string}"

        return xtb_status, message

    @staticmethod
    def assess_g16_calculations(calc: IsomerCollection) -> tuple[bool, str]:
        """
        Checks the status of a G16 calculations and
        returns True if not finished/converged correctly.

        Parameters
        ---------------------------
        calc
            The output of the calculation node

        Returns
        -------
        bool
            Boolean with status of g16 calculation
            True if it failed, False if it was succesful
        """

        g16_status = True
        message = f"G16 calculation failed. exit codes:"

        for i in range(len(calc.molecules)):
            isomer_obj = calc.molecules[i]
            isomer_exit_codes = json.loads(cast(str, calc.molecules[i].get_tag("g16_dft_exit_codes")))

            if 0 in isomer_exit_codes.values():
                g16_status = False
                message = "G16 finished correctly."
                isomer_rd_mol = isomer_obj._molecule
                isomer_structure = Loader.molecule_from_rdkit(isomer_rd_mol)

                g16_conf_geometries_tag = json.loads(cast(str, isomer_obj.get_tag("g16_dft_geometries")))
                n_conf_g16 = len(g16_conf_geometries_tag)
        
                message += f"\n\n There are {n_conf_g16} conformations for {isomer_obj.get_tag('logging_name')}."
                connectivity_string = f"{isomer_obj.get_tag('component')}\n"

                for idx in range(n_conf_g16):
                    g16_conf_json = g16_conf_geometries_tag[str(idx)]
                    
                    if isinstance(g16_conf_json, str):
                        message += f" {isomer_exit_codes.values()}. Gaussian Error Message: {g16_conf_json}. Check compchem_utils.py for reference of error messages."
                        continue
                    else:    
                        g16_conf_structure = Loader.molecule_from_json(g16_conf_json, f"conformer-{idx}")

                        parent_conf = calc.molecules[i].conformers[idx]
                        parent_conf_structure = Loader.molecule_from_conformer(parent_conf)
                        
                        if isomer_obj.has_tag('ar_carbon'):
                            
                            if check_collaps(g16_conf_structure, cast(int, isomer_obj.get_tag('ar_carbon')), cast(int, isomer_obj.get_tag('alk_carbon'))):
                                
                                message += f"conformer-{idx} of isomer {isomer_obj.get_tag('component')} collapsed to product"
                                del g16_conf_geometries_tag[str(idx)]
                            

                        if not check_connectivity(parent_conf_structure, g16_conf_structure):
                            message += f"Warning conformer-{idx} has changed connectivity compared to parent molecule. Please double check."
                                

                if len(g16_conf_geometries_tag) > 0:
                    n_conf_removed = n_conf_g16 - len(g16_conf_geometries_tag)
                    g16_status = False
                    message += f" {isomer_obj.get_tag('component')} {n_conf_removed} conformer(s) were removed for connectivity errors."
        
                else:
                    message += f" {isomer_exit_codes.values()}."
            else:
                message += f" {isomer_exit_codes.values()}."

        return g16_status, message

    def run(self) -> None:
        row = self.inp.receive()
        self.logger.info(f"type(row):{type(row)} row:{row}")
        self.get_generics(row)

         
        selected_reaction = getattr(template_repo, self.reaction.value)
        if hasattr(self, 'others'):
            self.logger.info(f"RSMI: {self.rsmi}")
            reaction = selected_reaction(self.id, self.rsmi, self.others, self.requested_components.value)
            
            self.logger.debug(f"Extra component of the reaction are: {self.others}")
        else:
            reaction = selected_reaction(self.id, self.rsmi, logger=self.logger)
        
        self.logger.info(f"Requested reaction mechanism for {self.reaction.value}")

        list_output = []
        for component in reaction.return_reaction_components():
            self.logger.info(component.component)
            list_output.append(component.to_isomer())

        final_out = []
        for iso_list in list_output:
            for iso in iso_list:
                final_out.append(IsomerCollection([iso]))

        for isomercollection in final_out:
            self.logger.info(
                f"Molecular Weight of {isomercollection.molecules[0].get_tag('component')} is {isomercollection.molecules[0].get_tag('mw')} "
            )

        self.logger.info(f"{len(final_out)} to be submitted to CREST node")
        self.out_crest.send(final_out)

        crest_results = self.inp_crest.receive()
        # These are results from Crest that we need to verify have completed succesfully
        
        self.logger.info(
            f"Received list of molecules from Crest node: {crest_results[0].molecules[0].name}"
        )
        crest_verified_complete: list[IsomerCollection] = []
        self.logger.info(
            f"{len(crest_results)} results to be verified for correctness"
        )

        n = 0
        while crest_results and n < 2:
            crest_to_resub = []
            crest_to_delete = []
            n += 1
            for crest_calc in crest_results:
                self.logger.info(
                    f"The isomer for {crest_calc.molecules[0].get_tag('component')} has {len(crest_calc.molecules[0].conformers)} conformers"
                )
                isomer_calc = crest_calc.molecules[0]
                if isomer_calc.has_tag('cc_carbons') and len(cast(NDArray[np.int32],isomer_calc.get_tag('cc_carbons')))>1:
                    carbon1 = int(cast(NDArray[np.int32], isomer_calc.get_tag('cc_carbons'))[0])
                    carbon2 = int(cast(NDArray[np.int32], isomer_calc.get_tag('cc_carbons'))[1])
                    self.logger.info("Checking if molecule has collapsed into the product ...")
                    best_conf = Loader.molecule_from_conformer(isomer_calc.conformers[0])
                    self.logger.info(check_collaps(best_conf, carbon1, carbon2))
                    
                    if check_collaps(best_conf, carbon1, carbon2):
                        self.logger.info(f"C1: {carbon1}, C2: {carbon2}")
                        crest_to_delete.append(crest_calc)
                        
                        self.logger.info(f"{crest_calc.molecules[0].get_tag('component')} collapsed to product. Discarded.")
                        continue
                    else:
                        self.logger.info(f"Did not find any collapsed structures for {crest_calc.molecules[0].get_tag('component')}.")   
                
            
                status, message = self.assess_crest_calculations(crest_calc)
                if status:
                    crest_to_resub.append(crest_calc)
                    self.logger.info(message)
                    self.logger.info(
                        f"The isomer has to be resubmitted, {len(crest_calc.molecules[0].conformers)} conformers left."
                    )
                    self.logger.info(f"{crest_calc} is resubmitted for iteration number {n}.")
                else:
                    crest_verified_complete.append(crest_calc)
                    self.logger.info(message)

                    self.logger.info(f"CREST for {crest_calc.molecules[0].get_tag('component')} is completed at iteration number {n}")
                    self.logger.info(
                        f"Correct calc, {crest_calc.molecules[0].get_tag('component')} has {len(crest_calc.molecules[0].conformers)} conformers"
                    )

            if crest_to_resub:
                self.out_crest.send(crest_to_resub)
                crest_results = self.inp_crest.receive()
                self.logger.info(f"{len(crest_results)} to resub to CREST")
            else:
                crest_results = []
                self.logger.info("Nothing to resubmit to CREST")

        self.logger.info(f"List of completed crest results: {crest_verified_complete}")
        self.out_xg16.send(crest_verified_complete)

        xg16_available_result_list = (
            self.inp_xg16.receive()
        )  # These are results from xTB and G16 that we need to verify have completed succesfully
        self.logger.info(
            f"Received list of molecules from xTB and Gaussian node: {xg16_available_result_list[0].molecules[0].name}"
        )
        xg16_verified_complete: list[IsomerCollection] = []
        self.logger.info(
            f"{len(xg16_available_result_list)} results to be verified for correctness"
        )


        m = 0
        while xg16_available_result_list:
            xg16_to_resub = []
            
            self.logger.info(f"calling the XTB-G16 sequence for round number {m + 1}.")
            self.logger.info(f"There are {len(xg16_available_result_list)} components left to converge.")
           
            for xg16_calc in xg16_available_result_list:
                status_xtb, message_xtb = self.assess_xtb_calculations(xg16_calc)
                status_g16, message_g16 = self.assess_g16_calculations(xg16_calc)

                isomer_calc = xg16_calc.molecules[0]
                
                if status_xtb and not status_g16:

                    self.logger.info(f"XTB: {message_xtb}.")
                    self.logger.info(
                        f"The {xg16_calc.molecules[0].get_tag('component')} showed a warning/error in XTB that wasn't confirmed by the DFT calculations."
                    )
                    xg16_verified_complete.append(xg16_calc)

                if status_g16 and not status_xtb:
                    conformers_tag = json.loads(
                        cast(str, xg16_calc.molecules[0].get_tag("g16_dft_geometries"))
                    )
                    n_conformers = len(conformers_tag)

                    self.logger.info(f"G16: {message_g16}.")
                    self.logger.info(
                        f"The {xg16_calc.molecules[0].get_tag('component')} has to be resubmitted: failed G16. {n_conformers} conformers left."
                    )
                    self.logger.info(f"{xg16_calc.molecules[0].get_tag('component')} is resubmitted for iteration number {m}.")
                    xg16_to_resub.append(xg16_calc)
                
                elif status_xtb and status_g16:
                    conformers_tag = json.loads(
                        cast(str, xg16_calc.molecules[0].get_tag("g16_dft_geometries"))
                    )
                    n_conformers = len(conformers_tag)

                    self.logger.info(f"XTB: {message_xtb}, G16: {message_g16}.")
                    self.logger.info(
                        f"The {xg16_calc.molecules[0].get_tag('component')} has failed both XTB and G16 calcs. {n_conformers} conformers left."
                    )
                    self.logger.info(f"{xg16_calc.molecules[0].get_tag('component')} is resubmitted for iteration number {m}.")
                    xg16_to_resub.append(xg16_calc)


                else:
                    xg16_verified_complete.append(xg16_calc)
                    self.logger.info(f"\n {message_xtb} {message_g16}") #readd 

            if xg16_to_resub and m < 0:

                self.out_xg16.send(xg16_to_resub)
                
                xg16_results = self.inp_xg16.receive()
                id_to_resub = []
                for comps in xg16_results:
                    id_to_resub.append(cast(str, comps.molecules[0].get_tag('component')))
                self.logger.info(f"received structures to re-run for: {', '.join(id_to_resub)}")
            
            elif xg16_to_resub and m >= 2:
                self.logger.info('Too many iterations. Loop will stop here. Please check the structures.')
                xg16_available_result_list = []
            else:
                xg16_available_result_list = []
                self.logger.info("no calc to resubmit to XTB-G16 nodes")
            m +=1


        self.out.send(xg16_verified_complete)



class TestSuiteReactionControl:
    @staticmethod
    def rdmol_to_jsontag(mol: Chem.rdchem.Mol) -> ConfTag:
        molecule = Loader.molecule_from_rdkit(mol)

        atoms: list[AtomEntry] = []
        for atom in molecule.atoms:
            atoms.append({"element": atom.label, "atom_id": atom.number, "coords": atom.position})

        conf_tag: ConfTag = {"atoms": atoms, "energy": 0.005, "gradient": 0.000}

        return conf_tag

    def test_full_wf(
        self,
        test_config: Any,
    ) -> None:
        rig = TestRig(ReactionControl, config=test_config)

        data = {
            "Libname": "EN13003-85",
            "Row": "0",
            "yield": "0.17733",
            "rsmi_processed": "Cc1ccc2c(Br)ccc(NC(=O)c3ccc(OC(C)C)cc3)c2n1.BrC1CCCC1>>Cc1ccc2c(C3CCCC3)ccc(NC(=O)c3ccc(OC(C)C)cc3)c2n1",
            "CorrectedReagentsSmiles": "(O=C([O-])[O-].[Na+].[Na+]).CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.C[Si](C)(C)[SiH]([Si](C)(C)C)[Si](C)(C)C.(CC(C)(C)c1ccnc(-c2cc(C(C)(C)C)ccn2)c1.F[P-](F)(F)(F)(F)F.Fc1cc(F)c(-c2ccc(C(F)(F)F)cn2)c([Ir+]c2cc(F)cc(F)c2-c2cc(C(F)(F)F)ccn2)c1).(COCCOC.Cl[Ni]Cl).COCCO",
        }

        input_reac = pd.Series(data)

        rd_mol = Chem.MolFromSmiles("OC(C)C")
        rd_mol = Chem.AddHs(rd_mol)
        AllChem.EmbedMolecule(rd_mol)

        correct = IsomerCollection([Isomer.from_rdmols([rd_mol])])

        rd_mol2 = Chem.MolFromSmiles("OC(C)C")
        rd_mol2 = Chem.AddHs(rd_mol2)
        AllChem.EmbedMolecule(rd_mol2)
        wrong = IsomerCollection([Isomer.from_rdmols([rd_mol2])])

        correct_conf1 = Chem.MolFromSmiles("OC(C)C")
        correct_conf1 = Chem.AddHs(correct_conf1)
        AllChem.EmbedMolecule(correct_conf1)

        wrong_conf2 = Chem.MolFromSmiles("OCCC")
        wrong_conf2 = Chem.AddHs(wrong_conf2)
        AllChem.EmbedMolecule(wrong_conf2)

        # create isomer and add a conformer for first test_molecule
        correct.molecules[0].set_tag('component', "Correct Molecule")
        correct.molecules[0].set_tag("crest_exit_code", 0)
        correct.molecules[0].clear_conformers()
        correct.molecules[0].add_conformer(
            Conformer.from_rdmol(
                correct_conf1, parent=correct.molecules[0], renumber=False, sanitize=False
            )
        )
        correct.molecules[0].set_tag("XTB_exit_codes", '{"0" : 0}')
        correct.molecules[0].set_tag("g16_exit_codes", '{"0" : 0}')
        correct.molecules[0].set_tag(
            "XTB_geometries",
            json.dumps({"0": [TestSuiteReactionControl.rdmol_to_jsontag(correct_conf1)]}),
        )

        wrong.molecules[0].set_tag('component', "Wrong Molecule")
        wrong.molecules[0].set_tag("crest_exit_code", 0)
        wrong.molecules[0].clear_conformers()
        wrong.molecules[0].add_conformer(
            Conformer.from_rdmol(
                wrong_conf2, parent=wrong.molecules[0], renumber=False, sanitize=False
            )
        )
        wrong.molecules[0].set_tag("XTB_exit_codes", '{"0" : 0}')
        wrong.molecules[0].set_tag("g16_exit_codes", '{"0" : 0}')
        wrong.molecules[0].set_tag(
            "XTB_geometries",
            json.dumps({"0": [TestSuiteReactionControl.rdmol_to_jsontag(wrong_conf2)]}),
        )

        res = rig.setup_run(
            inputs={
                "inp": [input_reac],
                "inp_crest": [[wrong], [correct]],
                "inp_xg16": [[wrong], [correct]],
            }
        )

        corr_out = res["out"].get()
        to_crest = res["out_crest"].get()
        to_xg16 = res["out_xg16"].get()

        assert corr_out is not None
        assert to_crest is not None
        assert to_xg16 is not None

        assert len(to_crest) == 16
        for mol in to_crest:
            assert mol.molecules[0].get_tag("mw") is not "0.0"

        assert len(to_xg16) == 1
        for mol in to_xg16:
            assert mol.molecules[0].get_tag("crest_exit_code") == 0
            assert mol.molecules[0].get_tag('component') == "Correct Molecule"

        assert len(corr_out) == 1
        for mol in corr_out:
            assert mol.molecules[0].get_tag('component') == "Correct Molecule"
            assert json.loads(mol.molecules[0].get_tag("XTB_exit_codes"))["0"] == 0
            assert json.loads(mol.molecules[0].get_tag("g16_exit_codes"))["0"] == 0
