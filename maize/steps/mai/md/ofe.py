"""OpenFE RBFE implementation"""

# pylint: disable=import-outside-toplevel, import-error

from collections import defaultdict
import copy
import csv
from dataclasses import dataclass
import itertools
import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Annotated, Any, Literal, Sequence, cast

import networkx as nx
import numpy as np
import pytest

from maize.core.node import Node
from maize.core.interface import Parameter, Flag, Suffix, Input, Output, FileParameter
from maize.utilities.chem import Isomer, IsomerCollection, load_sdf_library
from maize.utilities.chem.chem import save_sdf_library
from maize.utilities.testing import TestRig
from maize.utilities.io import Config


log = logging.getLogger(f"run-{os.getpid()}")


MappingType = Literal["star", "minimal", "minimal-redundant", "custom", "existing"]


# TODO figure out n_jobs params etc
def _parametrise_mols(
    mols: Sequence[Isomer], out_file: Path, n_cores: int = 2, n_workers: int = 1
) -> None:
    """Uses OpenFF bespokefit to parametrise small molecules"""
    from openff.bespokefit.workflows import BespokeWorkflowFactory
    from openff.bespokefit.executor import BespokeExecutor, BespokeWorkerConfig, wait_until_complete
    from openff.qcsubmit.common_structures import QCSpec
    from openff.toolkit.topology import Molecule
    from openff.toolkit import ForceField
    from openff.toolkit.utils.exceptions import ParameterLookupError

    ffmols = [Molecule.from_rdkit(mol._molecule) for mol in mols]
    spec = QCSpec(method="ani2x", basis=None, program="torchani", spec_name="ani2x")
    factory = BespokeWorkflowFactory(
        initial_force_field="openff-2.1.0.offxml", default_qc_specs=[spec]
    )
    log.info("Creating parametrisation schemas...")
    schemas = factory.optimization_schemas_from_molecules(ffmols, processors=n_cores * n_workers)

    # Do the fitting
    with BespokeExecutor(
        n_fragmenter_workers=1,
        n_optimizer_workers=1,
        n_qc_compute_workers=max(n_workers - 2, 1),
        qc_compute_worker_config=BespokeWorkerConfig(n_cores=n_cores),
    ) as exec:
        log.info("Fitting...")
        tasks = [exec.submit(input_schema=schema) for schema in schemas]
        outputs = [wait_until_complete(task) for task in tasks]

    # Combine FFs into single file, this is just following the `combine.py` CLI tool
    ffs: ForceField = []
    for i, out in enumerate(outputs):
        ff_file = f"mol-{i}.offxml"
        if out.bespoke_force_field is not None:
            out.bespoke_force_field.to_file(ff_file)
            ffs.append(ForceField(ff_file, load_plugins=True, allow_cosmetic_attributes=True))
        else:
            log.warning("Parameterisation failed for %s", mols[i])

    # Combine torsions
    master = copy.deepcopy(ffs[0])
    for ff in ffs[1:]:
        for parameter in ff.get_parameter_handler("ProperTorsions").parameters:
            try:
                master.get_parameter_handler("ProperTorsions")[parameter.smirks]
            except ParameterLookupError:
                master.get_parameter_handler("ProperTorsions").add_parameter(parameter=parameter)

    # Save combined FF
    master.to_file(filename=out_file, discard_cosmetic_attributes=True)


@dataclass
class FEPResult:
    smiles: tuple[str, str]
    ddg: float
    ddg_error: float
    mapping_score: float


class IndexingDict(dict[Any, int]):
    """Dictionary that converts each entry into a unique index"""

    def __getitem__(self, __key: Any) -> int:
        if __key not in self:
            super().__setitem__(__key, len(self))
        return super().__getitem__(__key)


EPS = 1e-2


class MakeAbsoluteMappingScore(Node):
    """Convert FEP mapping results to an absolute value"""

    tags = {"chemistry", "scorer", "tagger", "fep", "utility"}

    inp: Input[dict[tuple[str, str], FEPResult]] = Input()
    """FEP result input"""

    inp_mols: Input[list[IsomerCollection]] = Input()
    """Original molecules"""

    inp_ref: Input[Isomer] = Input(cached=True)
    """Reference molecule to compute absolute binding energies"""

    out: Output[list[IsomerCollection]] = Output()
    """Tagged mol output"""

    def run(self) -> None:
        results = self.inp.receive()
        mols = self.inp_mols.receive()
        ref = self.inp_ref.receive()

        isos = {iso.inchi: iso for mol in mols for iso in mol.molecules}
        if ref.inchi not in isos:
            isos[ref.inchi] = ref

        maps = defaultdict(list)
        for (a, b), res in results.items():
            maps[a].append(res.mapping_score)
            maps[b].append(res.mapping_score)

        for name, scores in maps.items():
            # We choose the median here to avoid bias in an example where we have a
            # subnetwork with universally good scores, and a single edge with a bad
            # score. In that case, one side of the edge will receive a bad score if
            # it's not forming other good edges, but the other side will show some
            # robustness to this and keep a good score (Thanks Annie!). We set nodes
            # with no edges to 0.
            isos[name].add_score("mapper", float(np.median(scores)) if scores else 0.0, agg="max")

        # In some cases, the edges defined in results might not cover
        # all input molecules (i.e. in the case of partial mapping
        # failures). In those cases we set the mapping score to 0.
        for iso in isos.values():
            if "mapper" not in iso.scores:
                iso.add_score("mapper", 0.0, agg="max")

        self.out.send(mols)


class MakeAbsolute(Node):
    """Convert FEP results to an absolute free energy"""

    tags = {"chemistry", "scorer", "tagger", "fep", "utility"}

    inp: Input[dict[tuple[str, str], FEPResult]] = Input()
    """FEP result input"""

    inp_mols: Input[list[IsomerCollection]] = Input()
    """Original molecules"""

    inp_ref: Input[Isomer] = Input(cached=True)
    """Reference molecule to compute absolute binding energies"""

    ref_score: Parameter[float] = Parameter(optional=True)
    """Reference score if not included as tag in reference mol (kJ/mol)"""

    out: Output[list[IsomerCollection]] = Output()
    """Tagged mol output"""

    def run(self) -> None:
        from cinnabar.stats import mle

        results = self.inp.receive()
        mols = self.inp_mols.receive()
        ref = self.inp_ref.receive()
        if ref.scored:
            ref_score = ref.primary_score
        elif self.ref_score.is_set:
            ref_score = self.ref_score.value
        else:
            ref_score = 0.0
        self.logger.info("Using reference %s with score of %s", ref.name or ref.inchi, ref_score)

        isos = {iso.inchi: iso for mol in mols for iso in mol.molecules}
        if ref.inchi not in isos:
            isos[ref.inchi] = ref

        # In some cases we might be calculating a partial network only, which can
        # make it impossible to provide real absolute BFE values. Considering usage
        # in RL loops, the safest thing is to just return NaNs and a warning.
        if ref.inchi not in set(itertools.chain(*results)):
            self.logger.warning(
                "Reference molecule '%s' not found in results or molecules (%s)",
                ref.name or ref.inchi,
                isos.keys(),
            )
            for iso in isos.values():
                iso.set_tag("fep", np.nan)
                iso.set_tag("fep_error", np.nan)
                iso.add_score_tag("fep")
            self.out.send(mols)
            return

        # Build graph to get maximum likelihood estimate
        graph = nx.DiGraph()

        # Use the same 'paranoid' approach as OpenFE and convert to indices (and back again)
        # https://github.com/OpenFreeEnergy/openfe/blob/main/openfecli/commands/gather.py
        neighbors = defaultdict(set)
        name2idx = IndexingDict()
        for (a, b), res in results.items():
            neighbors[a].add(b)
            neighbors[b].add(a)
            idx_a, idx_b = name2idx[a], name2idx[b]

            # A NaN result implies a failed edge and thus can cause the network
            # to be split into disconnected subgraphs. We will try to save the
            # campaign by using the subgraph including the reference later.
            if np.isnan(res.ddg):
                graph.add_node(idx_a)
                graph.add_node(idx_b)
                continue

            # MLE fails when the error is 0
            graph.add_edge(idx_a, idx_b, f_ij=res.ddg, f_dij=max(res.ddg_error, EPS))

        graph.nodes[name2idx[ref.inchi]]["f_i"] = ref_score
        graph.nodes[name2idx[ref.inchi]]["f_di"] = 0.1

        idx2name = {v: k for k, v in name2idx.items()}

        # Failed edges can partition the graph, in those cases we have no choice but to
        # only compute absolute values for the largest subgraph containing the reference
        if nx.number_weakly_connected_components(graph) > 1:
            node_lists = list(nx.weakly_connected_components(graph))

            # Find subgraph containing reference, set all other nodes to NaN
            for nodes in node_lists:
                if name2idx[ref.inchi] in nodes:
                    graph = graph.subgraph(nodes)
                    continue

                for node in nodes:
                    iso = isos[idx2name[node]]
                    iso.set_tag("fep", np.nan)
                    iso.set_tag("fep_error", 0.0)
                    iso.add_score_tag("fep")

        # Absolute FEs and covariance, we're only interested in the variance
        f_i, df_i = mle(graph, factor="f_ij", node_factor="f_i")
        df_i = np.sqrt(np.diagonal(df_i))

        for idx, dg, dg_err in zip(graph.nodes, f_i, df_i):
            iso = isos[idx2name[idx]]
            iso.set_tag("neighbors", list(neighbors[idx2name[idx]]))
            iso.set_tag("fep", dg)
            iso.set_tag("fep_error", dg_err)
            iso.add_score_tag("fep")
        self.out.send(mols)


class DynamicReference(Node):
    """Dynamically finds a reference molecule to optimise FEP map creation"""

    tags = {"chemistry", "fep", "utility"}

    inp: Input[list[Isomer]] = Input()
    """List of molecules to compute the ideal reference to"""

    inp_ref: Input[Isomer] = Input(cached=True)
    """Existing reference, will be used if no better reference can be found"""

    out: Output[Isomer] = Output()
    """Dynamic reference output"""

    pool: FileParameter[Annotated[Path, Suffix("sdf")]] = FileParameter(exist_required=False)
    """Library of existing molecules"""

    mapping: Parameter[MappingType] = Parameter(default="minimal")
    """Type of network to use for mapping"""

    mapping_backend: Parameter[Literal["lomap", "kartograf"]] = Parameter(default="lomap")
    """The mapping backend to use"""

    def run(self) -> None:
        import openfe
        import openfe.setup

        ref = self.inp_ref.receive()

        # With an empty pool (e.g. first epoch with FEP reinforcement
        # learning) we just send the existing reference
        if not self.pool.filepath.exists():
            self.logger.info(
                "Empty reference pool, using initial reference '%s'",
                ref.name or ref.inchi,
            )
            self.out.send(ref)
            return

        # Using a dynamic reference with a fixed mapping makes no sense
        if self.mapping.value in ("custom", "existing"):
            self.logger.warning(
                "Cannot use dynamic reference with mapping '%s', using initial reference '%s'",
                self.mapping.value,
                ref.name or ref.inchi,
            )
            self.out.send(ref)
            return

        existing = load_sdf_library(
            self.pool.filepath, split_strategy="none", renumber=False, sanitize=False
        )

        # Prepare mapper
        if self.mapping_backend.value == "kartograf":
            mapper = openfe.setup.KartografAtomMapper()
        else:
            mapper = openfe.LomapAtomMapper()
        scorer = openfe.lomap_scorers.default_lomap_score

        # Create mols to be mapped
        isomers = self.inp.receive()
        mols = [openfe.SmallMoleculeComponent(iso._molecule) for iso in isomers]

        # Create potential reference mols
        pool = [iso for mol in existing for iso in mol.molecules if np.isfinite(iso.primary_score)]
        pool.append(ref)
        pool_mols = [openfe.SmallMoleculeComponent(iso._molecule) for iso in pool]

        n_pool, n_mols = len(pool_mols), len(mols)
        dists = np.zeros((n_pool, n_mols))
        for i, pool_mol in enumerate(pool_mols):
            for j, mol in enumerate(mols):
                mappings = list(mapper.suggest_mappings(mol, pool_mol))
                if mappings:
                    dists[i, j] = scorer(mappings[0])

        # In a minimal mapping, the best network layout will be determined automatically,
        # thus OpenFE will choose the best edge to create to the reference (which will
        # generally be the edge with the best mapping score). That means we can simply
        # choose the molecule from the pool that maximizes the mapping score. In a star
        # map we want all edges to be good, so we have to choose the mean distance and
        # maximize that instead.
        if self.mapping.value == "star":
            new_ref = pool[dists.mean(axis=1).argmax(axis=0)]
        else:  # minimal, minimal-redundant
            new_ref = pool[dists.max(axis=1).argmax(axis=0)]

        self.logger.info(
            "Using reference '%s' with %s mapping (dg=%s)",
            new_ref.name or new_ref.inchi,
            self.mapping.value,
            new_ref.primary_score,
        )
        self.out.send(new_ref)


class SaveOpenFEResults(Node):
    """Save OpenFE result objects to a CSV"""

    tags = {"chemistry", "fep", "utility", "saving"}

    inp: Input[dict[tuple[str, str], FEPResult]] = Input()
    """FEP result input"""

    file: FileParameter[Annotated[Path, Suffix("csv")]] = FileParameter(exist_required=False)
    """Output CSV location"""

    def run(self) -> None:
        results = self.inp.receive()
        with self.file.filepath.open("w") as out:
            writer = csv.writer(out, delimiter=",")
            writer.writerow(["origin", "target", "origin-smiles", "target-smiles", "ddg", "error"])
            for (origin, target), edge in results.items():
                writer.writerow([origin, target, *edge.smiles, edge.ddg, edge.ddg_error])


class OpenAHFE(Node):
    """Run an absolute hydration free energy campaign using Open Free Energy"""

    tags = {"chemistry", "scorer", "tagger", "fep"}

    SM_FFS = Literal[
        "gaff-2.11", "openff-1.3.1", "openff-2.1.0", "espaloma-0.2.2", "smirnoff99Frosst-1.1.0"
    ]

    required_callables = ["openfe"]
    """
    openfe
        OpenFE executable, included with the package

    """

    required_packages = ["openfe"]
    """
    openfe
        OpenFE python package

    """

    inp: Input[list[Isomer]] = Input()
    """Molecule inputs"""

    out: Output[list[Isomer]] = Output()
    """Calculated edges"""

    temperature: Parameter[float] = Parameter(default=298.15)
    """Temperature in Kelvin to use for all simulations"""

    ion_concentration: Parameter[float] = Parameter(default=0.15)
    """Ion concentration in molar"""

    ion_pair: Parameter[tuple[str, str]] = Parameter(default=("Na+", "Cl-"))
    """Positive and negative ions to use for neutralization"""

    neutralize: Flag = Flag(default=True)
    """Whether to neutralize the system"""

    equilibration_length: Parameter[int] = Parameter(default=2000)
    """Length of equilibration simulation in ps"""

    production_length: Parameter[int] = Parameter(default=5000)
    """Length of production simulation in ps"""

    early_termination: Flag = Flag(default=False)
    """If ``True``, will terminate simulation when the MBAR error is below 0.12 kcal/mol"""

    molecule_forcefield: Parameter[str] = Parameter(default="openff-2.1.0")
    """
    Force field to use for the small molecule. For recommended options see
    :attr:`OpenFE.SM_FFS`, for all options :meth:`OpenFE.available_ffs`. If
    you want to use bespoke fitting using OpenFF-bespokefit, specify `bespoke`.

    """

    solvent: Parameter[Literal["tip3p", "spce", "tip4pew", "tip5p", "obc2", "gbn2"]] = Parameter(default="tip3p")
    """Water model to use"""

    padding: Parameter[float] = Parameter(default=1.2)
    """Minimum distance of the solute to the box edge"""

    sampler: Parameter[Literal["repex", "sams", "independent"]] = Parameter(default="repex")
    """Sampler to use"""

    n_repeats: Parameter[int] = Parameter(default=3)
    """Number of simulation repeats"""

    n_replicas: Parameter[int] = Parameter(default=11)
    """Number of replicas to use"""

    n_lambda: Parameter[int] = Parameter(default=11)
    """Number of lambda windows to use"""

    partial_charge_method: Parameter[Literal["am1bcc", "am1bccelf10", "nagl", "espaloma"]] = Parameter(default="am1bcc")
    """Partial charge assignment method to use"""

    endstate_dispersion_correction: Flag = Flag(default=False)
    """Whether to perform additional endstate dispersion correction"""

    dispersion_correction: Flag = Flag(default=False)
    """Whether to perform dispersion correction"""

    softcore_lj: Parameter[Literal["gapsys", "beutler"]] = Parameter(default="gapsys")
    """Whether to use softcore LJ by Gapsys or Beutler"""

    softcore_alpha: Parameter[float] = Parameter(default=0.85)
    """Softcore alpha parameter"""

    n_jobs: Parameter[int] = Parameter(default=1)
    """
    Number of calculations to perform simultaneously. Should be equal
    to the number of GPUs if local, or the number of batch submissions.

    """

    platform: Parameter[Literal["CUDA", "CPU", "OpenCL", "Reference"]] = Parameter(optional=True)
    """The OpenMM compute platform"""

    mapping_score_only: Flag = Flag(default=False)
    """If ``True``, will only return the mapping score and not run FEP"""

    trial: Flag = Flag(default=False)
    """
    If ``True``, will not run FEP and produce random values,
    for debugging and workflow testing purposes

    """

    @classmethod
    def available_ffs(cls) -> list[str]:
        """Lists all currently available small molecule force fields"""
        from openmmforcefields.generators import SystemGenerator

        return cast(list[str], SystemGenerator.SMALL_MOLECULE_FORCEFIELDS)

    def run(self) -> None:
        import openfe
        import openfe.setup
        import gufe
        from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
        from pint import DimensionalityError
        from openff.units import unit

        # Create molecules
        isomers = self.inp.receive()
        isos = {iso.name or iso.inchi: iso for iso in isomers}
        mols = {
            name: openfe.SmallMoleculeComponent.from_rdkit(iso._molecule)
            for name, iso in isos.items()
        }

        # Solvation
        p_ion, n_ion = self.ion_pair.value
        solvent = openfe.SolventComponent(
            positive_ion=p_ion,
            negative_ion=n_ion,
            neutralize=self.neutralize.value,
            ion_concentration=self.ion_concentration.value * unit.molar,
        )

        # Small molecule FF
        sm_ff = self.molecule_forcefield.value
        if sm_ff == "bespoke":
            _parametrise_mols(list(isos.values()), out_file=Path("bespoke.offxml"))
            sm_ff = "bespoke.offxml"

        # RBFE settings
        settings = AbsoluteSolvationProtocol.default_settings()
        settings.protocol_repeats = self.n_repeats.value
        settings.thermo_settings.temperature = self.temperature.value * unit.kelvin
        settings.solvent_engine_settings.compute_platform = (
            self.platform.value if self.platform.is_set else None
        )
        settings.solvent_forcefield_settings.small_molecule_forcefield = sm_ff
        settings.vacuum_forcefield_settings.small_molecule_forcefield = sm_ff
        settings.solvation_settings.solvent_model = self.solvent.value
        settings.solvation_settings.solvent_padding = self.padding.value * unit.nanometers
        settings.partial_charge_settings.partial_charge_method = self.partial_charge_method.value
        settings.solvent_simulation_settings.sampler_method = self.sampler.value
        settings.solvent_simulation_settings.equilibration_length = (
            self.equilibration_length.value * unit.picosecond
        )
        settings.solvent_simulation_settings.production_length = (
            self.production_length.value * unit.picosecond
        )
        settings.vacuum_simulation_settings.sampler_method = self.sampler.value
        settings.vacuum_simulation_settings.equilibration_length = (
            self.equilibration_length.value * unit.picosecond
        )
        settings.vacuum_simulation_settings.production_length = (
            self.production_length.value * unit.picosecond
        )
        if self.early_termination.value:
            settings.solvent_simulation_settings.early_termination_target_error = (
                0.12 * unit.kilocalorie_per_mole
            )
            settings.vacuum_simulation_settings.early_termination_target_error = (
                0.12 * unit.kilocalorie_per_mole
            )
        protocol = AbsoluteSolvationProtocol(settings)

        # Setup transforms
        self.logger.info("Generating transforms")
        transforms: dict[str, gufe.tokenization.GufeTokenizable] = {}
        for name, mol in mols.items():
            a = openfe.ChemicalSystem({"ligand": mol, "solvent": solvent}, name=name)
            b = openfe.ChemicalSystem({"solvent": solvent})
            transforms[name] = openfe.Transformation(
                stateA=a,
                stateB=b,
                mapping=None,
                protocol=protocol,
                name=f"ahfe_{a.name}",
            )

        # Prepare commands
        commands = []
        for name, transform in transforms.items():
            # Only run required edges
            if not (res_file := Path(f"{name}_res.json")).exists():
                tf_dir = Path(f"tf-{name}")
                tf_dir.mkdir()
                tf_json = tf_dir / f"{name}.json"
                transform.dump(tf_json)
                commands.append(
                    f"{self.runnable['openfe']} quickrun -d {tf_dir.as_posix()} "
                    f"-o {res_file.as_posix()} {tf_json.as_posix()}"
                )

        # Run
        use_mps = (
            self.platform.is_set and self.platform.value == "CUDA" and self.batch_options.is_set
        )
        self.logger.info("Running %s transforms", len(transforms))

        if not self.trial.value:
            self.run_multi(
                commands,
                n_jobs=self.n_jobs.value,
                raise_on_failure=False,
                cuda_mps=use_mps,
            )
        else:
            self.logger.warning("Running in trial mode, generating fake results!")

        def fail(iso: Isomer) -> None:
            iso.set_tag("ahfe", np.nan)
            iso.set_tag("ahfe_error", 0.0)
            iso.add_score_tag("ahfe", agg="min")

        # Parse results
        msg = "Parsing results"
        for name, transform in transforms.items():
            iso = isos[name]

            # Catch failed edges
            if not Path(f"{name}_res.json").exists():
                fail(iso)
                msg += f"\n  {name}:  failed (no result)"
                continue

            with Path(f"{name}_res.json").open("r") as res:
                try:
                    data = json.load(res, cls=gufe.tokenization.JSON_HANDLER.decoder)
                except (json.JSONDecodeError, DimensionalityError) as err:
                    fail(iso)
                    self.logger.warning(
                        "Error parsing %s (Error: %s)", f"{name}_res.json", err
                    )
                    continue

            # dG + error
            if data["estimate"] is None:
                fail(iso)
                msg += f"\n  {name}:  failed (NaN)"
                continue

            ahfe = data["estimate"].magnitude
            ahfe_unc = data["uncertainty"].magnitude

            iso.set_tag("ahfe", ahfe)
            iso.set_tag("ahfe_error", ahfe_unc)
            iso.add_score_tag("ahfe", agg="min")
            msg += f"\n  {name}:  dG_solv={ahfe:4.4f} kcal/mol  err={ahfe_unc:4.4f}"

        self.logger.info(msg)

        self.out.send(list(isos.values()))


class OpenRFE(Node):
    """Run an RBFE campaign using Open Free Energy"""

    tags = {"chemistry", "scorer", "tagger", "fep"}

    SM_FFS = Literal[
        "gaff-2.11", "openff-1.3.1", "openff-2.1.0", "espaloma-0.2.2", "smirnoff99Frosst-1.1.0"
    ]

    required_callables = ["openfe"]
    """
    openfe
        OpenFE executable, included with the package

    """

    required_packages = ["openfe"]
    """
    openfe
        OpenFE python package

    """

    inp: Input[list[Isomer]] = Input()
    """Molecule inputs"""

    inp_ref: Input[Isomer] = Input(optional=True)
    """Reference molecule input for star-maps"""

    inp_protein: Input[Annotated[Path, Suffix("pdb")]] = Input(cached=True)
    """Protein structure"""

    inp_cofactor: Input[Isomer] = Input(optional=True)
    """Optional cofactor input"""

    out: Output[dict[tuple[str, str], FEPResult]] = Output()
    """Calculated edges"""

    continue_from: FileParameter[Path] = FileParameter(optional=True)
    """A folder containing a partially-run OpenFE campaign to continue from"""

    dump_to: FileParameter[Path] = FileParameter(optional=True)
    """A folder to dump all generated data to"""

    mapping: Parameter[MappingType] = Parameter(default="minimal")
    """Type of network to use for mapping"""

    mapping_backend: Parameter[Literal["lomap", "kartograf"]] = Parameter(default="lomap")
    """The mapping backend to use"""

    network: FileParameter[Annotated[Path, Suffix("edge")]] = FileParameter(optional=True)
    """An optional alternative FEPMapper atom mapping file, use ``mapping = "custom"``"""

    temperature: Parameter[float] = Parameter(default=298.15)
    """Temperature in Kelvin to use for all simulations"""

    ion_concentration: Parameter[float] = Parameter(default=0.15)
    """Ion concentration in molar"""

    ion_pair: Parameter[tuple[str, str]] = Parameter(default=("Na+", "Cl-"))
    """Positive and negative ions to use for neutralization"""

    neutralize: Flag = Flag(default=True)
    """Whether to neutralize the system"""

    equilibration_length: Parameter[int] = Parameter(default=2000)
    """Length of equilibration simulation in ps"""

    production_length: Parameter[int] = Parameter(default=5000)
    """Length of production simulation in ps"""

    early_termination: Flag = Flag(default=False)
    """If ``True``, will terminate simulation when the MBAR error is below 0.12 kcal/mol"""

    molecule_forcefield: Parameter[str] = Parameter(default="openff-2.1.0")
    """
    Force field to use for the small molecule. For recommended options see
    :attr:`OpenFE.SM_FFS`, for all options :meth:`OpenFE.available_ffs`. If
    you want to use bespoke fitting using OpenFF-bespokefit, specify `bespoke`.

    """

    solvent: Parameter[Literal["tip3p", "spce", "tip4pew", "tip5p", "obc2", "gbn2"]] = Parameter(default="tip3p")
    """Water model to use"""

    padding: Parameter[float] = Parameter(default=1.2)
    """Minimum distance of the solute to the box edge"""

    sampler: Parameter[Literal["repex", "sams", "independent"]] = Parameter(default="repex")
    """Sampler to use"""

    n_repeats: Parameter[int] = Parameter(default=3)
    """Number of simulation repeats"""

    n_replicas: Parameter[int] = Parameter(default=11)
    """Number of replicas to use"""

    n_lambda: Parameter[int] = Parameter(default=11)
    """Number of lambda windows to use"""

    partial_charge_method: Parameter[Literal["am1bcc", "am1bccelf10", "nagl", "espaloma"]] = Parameter(default="am1bcc")
    """Partial charge assignment method to use"""

    endstate_dispersion_correction: Flag = Flag(default=False)
    """Whether to perform additional endstate dispersion correction"""

    dispersion_correction: Flag = Flag(default=False)
    """Whether to perform dispersion correction"""

    softcore_lj: Parameter[Literal["gapsys", "beutler"]] = Parameter(default="gapsys")
    """Whether to use softcore LJ by Gapsys or Beutler"""

    softcore_alpha: Parameter[float] = Parameter(default=0.85)
    """Softcore alpha parameter"""

    n_jobs: Parameter[int] = Parameter(default=1)
    """
    Number of calculations to perform simultaneously. Should be equal
    to the number of GPUs if local, or the number of batch submissions.

    """

    platform: Parameter[Literal["CUDA", "CPU", "OpenCL", "Reference"]] = Parameter(optional=True)
    """The OpenMM compute platform"""

    mapping_score_only: Flag = Flag(default=False)
    """If ``True``, will only return the mapping score and not run FEP"""

    trial: Flag = Flag(default=False)
    """
    If ``True``, will not run FEP and produce random values,
    for debugging and workflow testing purposes

    """

    @classmethod
    def available_ffs(cls) -> list[str]:
        """Lists all currently available small molecule force fields"""
        from openmmforcefields.generators import SystemGenerator

        return cast(list[str], SystemGenerator.SMALL_MOLECULE_FORCEFIELDS)

    def run(self) -> None:
        import openfe
        import openfe.setup
        import gufe
        from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
        from pint import DimensionalityError
        from openff.units import unit

        # Explicit or implicit solvent
        implicit = self.solvent.value in ("obc2", "gbn2")

        # Prepare mapper
        if self.mapping_backend.value == "kartograf":
            mapper = openfe.setup.KartografAtomMapper()
        else:
            mapper = openfe.LomapAtomMapper()
        scorer = openfe.lomap_scorers.default_lomap_score

        # Create molecules
        isomers = self.inp.receive()
        if not self.mapping.value == "custom":
            for iso in isomers:
                iso.name = iso.inchi

        isos = {iso.name: iso for iso in isomers}
        mols = {
            name: openfe.SmallMoleculeComponent.from_rdkit(iso._molecule)
            for name, iso in isos.items()
        }

        # Generate network
        self.logger.info("Generating '%s' network", self.mapping.value)
        if self.mapping.value == "star":
            isomer_ref = self.inp_ref.receive()
            isomer_ref.name = isomer_ref.inchi
            ref = openfe.SmallMoleculeComponent.from_rdkit(isomer_ref._molecule)
            planner = openfe.ligand_network_planning.generate_radial_network
            network = planner(mols.values(), central_ligand=ref, mappers=[mapper], scorer=scorer)

            # Add reference to library for later lookup
            isos[isomer_ref.name] = isomer_ref

        elif self.mapping.value in ("minimal", "minimal-redundant"):
            mols_to_map = mols.copy()
            if (opt_ref := self.inp_ref.receive_optional()) is not None:
                opt_ref.name = opt_ref.inchi
                ref = openfe.SmallMoleculeComponent.from_rdkit(opt_ref._molecule, name=opt_ref.name)
                self.logger.info("Using '%s' as a reference", opt_ref.name)
                mols_to_map[opt_ref.name] = ref

                # Add reference to library for later lookup
                isos[opt_ref.name] = opt_ref

            if self.mapping.value == "minimal":
                planner = openfe.ligand_network_planning.generate_minimal_spanning_network
            else:
                planner = openfe.ligand_network_planning.generate_minimal_redundant_network

            try:
                network = planner(list(mols_to_map.values()), mappers=[mapper], scorer=scorer)

            # This is a bit of a hack, it would be nicer if we could just get a list
            # of failed mappings as an additional return value. For now we just catch
            # the error, inform the user about missing edges, and try again.
            except RuntimeError as err:
                missing = re.findall(r"name=([A-Z\-]*)", err.args[0])
                if not missing:
                    raise
                for missing_mol in missing:
                    self.logger.warning("Mapper was unable to create an edge for '%s'", missing_mol)
                    mols_to_map.pop(missing_mol)
                network = planner(list(mols_to_map.values()), mappers=[mapper], scorer=scorer)

        elif self.mapping.value == "custom" and self.network.is_set:
            network = openfe.ligand_network_planning.load_fepplus_network(
                ligands=mols.values(), mapper=mapper, network_file=self.network.filepath
            )
        elif self.mapping.value == "existing" and self.continue_from.is_set:
            with (self.continue_from.filepath / "network.graphml").open("r") as file:
                network = openfe.LigandNetwork.from_graphml(file.read())

        msg = "Created network with following mappings:"
        for mapping in network.edges:
            msg += (
                f"\n  {mapping.componentA.name}-{mapping.componentB.name}"
                f"  score={scorer(mapping):4.4f}"
            )
        self.logger.info(msg)

        # Save the network for reference / reuse
        with Path("network.graphml").open("w") as file:
            file.write(network.to_graphml())

        # Solvation
        p_ion, n_ion = self.ion_pair.value
        solvent = openfe.SolventComponent(
            positive_ion=p_ion,
            negative_ion=n_ion,
            neutralize=self.neutralize.value,
            ion_concentration=self.ion_concentration.value * unit.molar,
        )

        # Cofactor
        cofactor = None
        if (opt_cof := self.inp_cofactor.receive_optional()) is not None:
            cofactor = openfe.SmallMoleculeComponent.from_rdkit(opt_cof._molecule)

        # Receptor
        protein_file = self.inp_protein.receive()
        protein = openfe.ProteinComponent.from_pdb_file(protein_file.as_posix())

        # Mapping score only, useful for generative model warmup
        if self.mapping_score_only.value:
            self.logger.info("Only calculating mapper score")
            mappings = {}
            for mapping in network.edges:
                a, b = mapping.componentA.name, mapping.componentB.name
                mappings[(a, b)] = FEPResult(
                    smiles=(isos[a].to_smiles(remove_h=True), isos[b].to_smiles(remove_h=True)),
                    ddg=np.nan,
                    ddg_error=np.nan,
                    mapping_score=scorer(mapping),
                )
            self.out.send(mappings)
            return

        # Small molecule FF
        sm_ff = self.molecule_forcefield.value
        if sm_ff == "bespoke":
            _parametrise_mols(list(isos.values()), out_file=Path("bespoke.offxml"))
            sm_ff = "bespoke.offxml"

        # RBFE settings
        settings = RelativeHybridTopologyProtocol.default_settings()
        settings.protocol_repeats = self.n_repeats.value
        settings.thermo_settings.temperature = self.temperature.value * unit.kelvin
        settings.engine_settings.compute_platform = (
            self.platform.value if self.platform.is_set else None
        )
        if implicit:
            settings.forcefield_settings.forcefields = ["amber/ff14SB.xml", "amber/phosaa10.xml", f"implicit/{self.solvent.value}.xml"]
            settings.forcefield_settings.nonbonded_method = "NoCutoff"
        settings.forcefield_settings.small_molecule_forcefield = sm_ff
        settings.lambda_settings.lambda_windows = self.n_lambda.value
        if not implicit:
            settings.solvation_settings.solvent_model = self.solvent.value
            settings.solvation_settings.solvent_padding = self.padding.value * unit.nanometers
        settings.partial_charge_settings.partial_charge_method = self.partial_charge_method.value
        settings.alchemical_settings.endstate_dispersion_correction = self.endstate_dispersion_correction.value
        settings.alchemical_settings.use_dispersion_correction = self.dispersion_correction.value
        settings.alchemical_settings.softcore_LJ = self.softcore_lj.value
        settings.alchemical_settings.softcore_alpha = self.softcore_alpha.value
        settings.simulation_settings.sampler_method = self.sampler.value
        settings.simulation_settings.n_replicas = self.n_replicas.value
        settings.simulation_settings.equilibration_length = (
            self.equilibration_length.value * unit.picosecond
        )
        settings.simulation_settings.production_length = (
            self.production_length.value * unit.picosecond
        )
        if self.early_termination.value:
            settings.simulation_settings.early_termination_target_error = (
                0.12 * unit.kilocalorie_per_mole
            )
        protocol = RelativeHybridTopologyProtocol(settings)

        # Setup transforms
        self.logger.info("Generating transforms")
        transforms: list[dict[str, gufe.tokenization.GufeTokenizable]] = []
        for mapping in network.edges:
            dags: dict[str, gufe.tokenization.GufeTokenizable] = {}

            # Filter out self-mappings (charge changes are not allowed at this time)
            if mapping.componentA.name == mapping.componentB.name:
                self.logger.warning(
                    "Cannot run edge between identical components ('%s')", mapping.componentA.name
                )
                continue

            for leg in ("solvent", "complex"):
                a_setup = {"ligand": mapping.componentA}
                b_setup = {"ligand": mapping.componentB}

                if not implicit:
                    a_setup["solvent"] = solvent
                    b_setup["solvent"] = solvent

                if leg == "complex":
                    a_setup["protein"] = protein
                    b_setup["protein"] = protein

                if cofactor is not None:
                    a_setup["cofactor"] = cofactor
                    b_setup["cofactor"] = cofactor

                a = openfe.ChemicalSystem(a_setup, name=f"{mapping.componentA.name}_{leg}")
                b = openfe.ChemicalSystem(b_setup, name=f"{mapping.componentB.name}_{leg}")
                transform = openfe.Transformation(
                    stateA=a,
                    stateB=b,
                    mapping={"ligand": mapping},
                    protocol=protocol,
                    name=f"rbfe_{a.name}_{b.name}_{leg}",
                )
                dags[leg] = transform
            transforms.append(dags)

        if self.continue_from.is_set:
            for res_file in self.continue_from.filepath.glob("*_res.json"):
                self.logger.info("Found existing result, copying %s", res_file)
                shutil.copy(res_file, Path())

        # Prepare commands
        commands = []
        for dags in transforms:
            for transform in dags.values():
                # Only run required edges
                if not (res_file := Path(f"{transform.name}_res.json")).exists():
                    tf_dir = Path(f"tf-{transform.name}")
                    tf_dir.mkdir()
                    tf_json = tf_dir / f"{transform.name}.json"
                    transform.dump(tf_json)
                    commands.append(
                        f"{self.runnable['openfe']} quickrun -d {tf_dir.as_posix()} "
                        f"-o {res_file.as_posix()} {tf_json.as_posix()}"
                    )

        # Run
        use_mps = (
            self.platform.is_set and self.platform.value == "CUDA" and self.batch_options.is_set
        )
        self.logger.info("Running %s transforms", 2 * len(transforms))

        if not self.trial.value:
            self.run_multi(
                commands,
                n_jobs=self.n_jobs.value,
                raise_on_failure=False,
                cuda_mps=use_mps,
            )
        else:
            self.logger.warning("Running in trial mode, generating fake results!")

        def _failed_edge(
            a: str, b: str, test_data: bool = False, mapping_score: float = np.nan
        ) -> FEPResult:
            return FEPResult(
                ddg=np.random.normal(scale=2) if test_data else np.nan,
                ddg_error=np.random.random() if test_data else np.nan,
                smiles=(isos[a].to_smiles(remove_h=True), isos[b].to_smiles(remove_h=True)),
                mapping_score=mapping_score,
            )

        # Parse results
        msg = "Parsing results"
        results = {}
        for dags in transforms:
            data = {}

            transform = list(dags.values())[0]
            a = transform.stateA.name.removesuffix("_solvent")
            b = transform.stateB.name.removesuffix("_solvent")

            # Catch failed edges
            if any(not Path(f"{tf.name}_res.json").exists() for tf in dags.values()):
                msg += f"\n  {a} -> {b}:  failed (no result)"
                results[(a, b)] = _failed_edge(
                    a,
                    b,
                    test_data=self.trial.value,
                    mapping_score=scorer(transform.mapping[0]),
                )
                continue

            parsing_error = False
            for leg, transform in dags.items():
                with Path(f"{transform.name}_res.json").open("r") as res:
                    try:
                        data[leg] = json.load(res, cls=gufe.tokenization.JSON_HANDLER.decoder)
                    except (json.JSONDecodeError, DimensionalityError) as err:
                        parsing_error = True
                        self.logger.warning(
                            "Error parsing %s (Error: %s)", f"{transform.name}_res.json", err
                        )

            if parsing_error:
                msg += f"\n  {a} -> {b}:  failed (result parsing)"
                results[(a, b)] = _failed_edge(
                    a, b, mapping_score=scorer(transform.mapping[0])
                )
                continue

            # Legs
            dat_complex = data["complex"]["estimate"]
            dat_solvent = data["solvent"]["estimate"]

            # Catch failed edges
            if any(leg is None for leg in (dat_complex, dat_solvent)):
                msg += f"\n  {a} -> {b}:  failed (cmpl={dat_complex}, solv={dat_solvent})"
                results[(a, b)] = _failed_edge(
                    a, b, mapping_score=scorer(transform.mapping[0])
                )
                continue

            # Compute ddG + error
            ddg = dat_complex - dat_solvent
            complex_err, solvent_err = (
                data["complex"]["uncertainty"],
                data["solvent"]["uncertainty"],
            )
            ddg_err = np.sqrt(complex_err**2 + solvent_err**2 - 2 * complex_err * solvent_err)
            msg += f"\n  {a} -> {b}:  ddG={ddg:4.4f}  err={ddg_err:4.4f}"
            results[(a, b)] = FEPResult(
                ddg=ddg.magnitude,
                ddg_error=ddg_err.magnitude,
                smiles=(isos[a].to_smiles(remove_h=True), isos[b].to_smiles(remove_h=True)),
                mapping_score=scorer(transform.mapping[0]),
            )
        self.logger.info(msg)

        # Move results + raw data to dumping location
        if self.dump_to.is_set:
            dump_folder = self.dump_to.value
            dump_folder.mkdir(exist_ok=True)
            for folder in Path().glob("tf-*"):
                if not (dump_folder / folder.name).exists():
                    shutil.move(folder, dump_folder)
            for res_file in Path().glob("*_res.json"):
                if not (dump_folder / res_file.name).exists():
                    shutil.copy(res_file, dump_folder)
            shutil.copy(Path("network.graphml"), dump_folder)

        self.out.send(results)


# 1UYD previously published with Icolos (IcolosData/molecules/1UYD)
@pytest.fixture
def protein_path(shared_datadir: Path) -> Path:
    return shared_datadir / "tnks.pdb"


@pytest.fixture
def ligand_path(shared_datadir: Path) -> Path:
    return shared_datadir / "target.sdf"


@pytest.fixture
def ref_path(shared_datadir: Path) -> Path:
    return shared_datadir / "ref.sdf"


@pytest.fixture
def bad_ligand_path(shared_datadir: Path) -> Path:
    return shared_datadir / "nec-docked.sdf"


@pytest.fixture
def result_network() -> tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]]:
    isos = {smi: Isomer.from_smiles(smi) for smi in ["C", "CC", "CCC", "CCCO"]}
    mols = [IsomerCollection([iso]) for iso in isos.values()]
    return mols, {
        (isos["C"].inchi, isos["CC"].inchi): FEPResult(
            smiles=("C", "CC"), ddg=-2.0, ddg_error=0.5, mapping_score=0.8
        ),
        (isos["CC"].inchi, isos["CCC"].inchi): FEPResult(
            smiles=("CC", "CCC"), ddg=1.0, ddg_error=0.2, mapping_score=0.9
        ),
        (isos["CCC"].inchi, isos["CCCO"].inchi): FEPResult(
            smiles=("CCC", "CCCO"), ddg=0.5, ddg_error=0.8, mapping_score=0.8
        ),
        (isos["CCCO"].inchi, isos["CC"].inchi): FEPResult(
            smiles=("CCCO", "CC"), ddg=-1.3, ddg_error=0.1, mapping_score=0.5
        ),
    }


@pytest.fixture
def result_network_sub(
    result_network: tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]],
) -> tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]]:
    mols, net = result_network
    c, cc, *_ = mols
    net[(c.molecules[0].inchi, cc.molecules[0].inchi)] = FEPResult(
        smiles=("C", "CC"), ddg=np.nan, ddg_error=0.1, mapping_score=0.5
    )
    return mols, net


class TestSuiteOpenFE:
    @pytest.mark.needs_node("openahfe")
    def test_OpenAHFE(
        self,
        temp_working_dir: Path,
        test_config: Config,
        ligand_path: Path,
    ) -> None:
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenAHFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]]},
            parameters={
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
            },
        )
        ret = res["out"].get()
        assert ret is not None
        for iso in ret:
            assert np.isfinite(iso.scores["ahfe"])
            assert np.isfinite(iso.get_tag("ahfe_error"))

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol, ref]], "inp_protein": [protein_path]},
            parameters={
                "mapping": "minimal",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
            },
        )
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert edge.ddg
            assert edge.ddg_error

    # See https://github.com/OpenFreeEnergy/openfe/blob/27c91161a7f24aac6b1c51b7dfa41b3c107d142e/openfe/protocols/openmm_rfe/_rfe_utils/relative.py#L309
    @pytest.mark.needs_node("openrfe")
    @pytest.mark.xfail(reason="CustomGBForce is not whitelisted by OpenFE, maybe it can be added?")
    def test_OpenRFE_implicit(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol, ref]], "inp_protein": [protein_path]},
            parameters={
                "mapping": "minimal",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "solvent": "gbn2",
            },
        )
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert edge.ddg
            assert edge.ddg_error

    @pytest.mark.needs_node("openrfe")
    @pytest.mark.skip(reason="Bespokefit is not ready yet")
    def test_OpenRFE_bespoke(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol, ref]], "inp_protein": [protein_path]},
            parameters={
                "mapping": "minimal",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "molecule_forcefield": "bespoke",
            },
        )
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert edge.ddg
            assert edge.ddg_error

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
            },
        )
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert edge.ddg
            assert edge.ddg_error

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star_map_only(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "mapping_score_only": True,
            },
        )
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isnan(edge.ddg)
            assert np.isnan(edge.ddg_error)
            assert 0 <= edge.mapping_score <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star_trial(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
            },
        )
        assert Path(
            "tf-rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex/rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_minimal_trial(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "minimal",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
            },
        )
        assert Path(
            "tf-rbfe_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_complex/rbfe_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_minimal_redundant_trial(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "minimal-redundant",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
            },
        )
        assert Path(
            "tf-rbfe_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_complex/rbfe_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_minimal_bad_trial(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        bad_ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol1 = Isomer.from_sdf(ligand_path)
        mol2 = Isomer.from_sdf(bad_ligand_path)
        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol1, mol2]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "minimal",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
            },
        )
        assert Path(
            "tf-rbfe_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_complex/rbfe_UZYTVPMNMQLENF-QWOVJGMINA-N_complex_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        assert len(edges) == 1
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star_trial_cont(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol1 = Isomer.from_sdf(ligand_path)
        mol2 = Isomer.from_sdf(ligand_path)
        existing = Path("./existing")
        existing.mkdir()
        file = existing / (
            "rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_"
            "UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex_res.json"
        )
        file.touch()

        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol1, mol2]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
                "continue_from": existing,
            },
        )
        assert not Path(
            "transforms/rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_"
            "UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_trial_cont_map(
        self,
        shared_datadir: Path,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        existing = Path("./existing")
        existing.mkdir()
        file = existing / (
            "rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_"
            "UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex_res.json"
        )
        file.touch()
        shutil.copy(shared_datadir / "network.graphml", existing)

        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol, ref]], "inp_protein": [protein_path]},
            parameters={
                "mapping": "existing",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
                "continue_from": existing,
            },
        )
        assert not Path(
            "transforms/rbfe_ILBZVTXOJSVUIM-QWOVJGMINA-N_complex_"
            "UZYTVPMNMQLENF-QWOVJGMINA-N_complex_complex.json"
        ).exists()
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    @pytest.mark.needs_node("openrfe")
    def test_OpenRFE_star_trial_dump(
        self,
        temp_working_dir: Path,
        test_config: Config,
        protein_path: Path,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        dump = Path("dump")
        dump.mkdir()

        rig = TestRig(OpenRFE, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [[mol]], "inp_ref": [ref], "inp_protein": [protein_path]},
            parameters={
                "mapping": "star",
                "equilibration_length": 5,
                "production_length": 10,
                "n_repeats": 2,
                "n_replicas": 4,
                "n_lambda": 4,
                "trial": True,
                "dump_to": dump,
            },
        )
        assert len(list(dump.iterdir())) == 3
        edges = res["out"].get()
        assert edges is not None
        for _, edge in edges.items():
            assert np.isfinite(edge.ddg)
            assert 0 <= edge.ddg_error <= 1

    def test_DynamicReference(
        self,
        tmp_path: Path,
        test_config: Config,
    ) -> None:
        mols = []
        for i in range(1, 4):
            iso = Isomer.from_smiles("C" * i)
            iso.add_score("score", 1.0, agg="max")
            iso.embed()
            mols.append(IsomerCollection([iso]))
        save_sdf_library(tmp_path / "pool.sdf", mols)

        ref = Isomer.from_smiles("CCCCCO")
        ref.add_score("score", 2.0, agg="max")
        ref.embed()
        isos = [Isomer.from_smiles("C" * i) for i in range(4, 6)]
        for iso in isos:
            iso.embed()

        rig = TestRig(DynamicReference, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [isos], "inp_ref": [ref]}, parameters={"pool": tmp_path / "pool.sdf"}
        )
        new_ref = res["out"].get()
        assert new_ref is not None
        assert new_ref.to_smiles(remove_h=True) == "CCC"

        rig = TestRig(DynamicReference, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [isos], "inp_ref": [ref]}, parameters={"pool": tmp_path / "newpool.sdf"}
        )
        new_ref = res["out"].get()
        assert new_ref is not None
        assert new_ref.to_smiles(remove_h=True) == "CCCCCO"

    def test_MakeAbsoluteMappingScore(
        self,
        tmp_path: Path,
        test_config: Config,
        result_network: tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]],
    ) -> None:
        ref = Isomer.from_smiles("C")
        mols, data = result_network
        rig = TestRig(MakeAbsoluteMappingScore, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.allclose(isos["C"].scores["mapper"], 0.8, 0.1)
        assert np.allclose(isos["CC"].scores["mapper"], 0.8, 0.1)
        assert np.allclose(isos["CCC"].scores["mapper"], 0.8, 0.1)
        assert np.allclose(isos["CCCO"].scores["mapper"], 0.65, 0.1)

    def test_MakeAbsolute(
        self,
        tmp_path: Path,
        test_config: Config,
        result_network: tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]],
    ) -> None:
        ref = Isomer.from_smiles("C")
        mols, data = result_network
        rig = TestRig(MakeAbsolute, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
            parameters={"ref_score": -10.0},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.allclose(isos["C"].scores["fep"], -10.0, 0.1)
        assert np.allclose(isos["CC"].scores["fep"], -12.0, 0.1)
        assert np.allclose(isos["CCC"].scores["fep"], -11.01, 0.1)
        assert np.allclose(isos["CCCO"].scores["fep"], -10.69, 0.1)
        assert isos["C"].get_tag("neighbors") == [isos["CC"].inchi]
        assert set(isos["CC"].get_tag("neighbors")) == {
            isos["C"].inchi,
            isos["CCC"].inchi,
            isos["CCCO"].inchi,
        }

        ref = Isomer.from_smiles("CC")
        mols, data = result_network
        rig = TestRig(MakeAbsolute, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
            parameters={"ref_score": -10.0},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.allclose(isos["C"].scores["fep"], -8.0, 0.1)
        assert np.allclose(isos["CC"].scores["fep"], -10.0, 0.1)
        assert np.allclose(isos["CCC"].scores["fep"], -9.01, 0.1)
        assert np.allclose(isos["CCCO"].scores["fep"], -8.69, 0.1)

        ref = Isomer.from_smiles("CCCCC")
        mols, data = result_network
        rig = TestRig(MakeAbsolute, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
            parameters={"ref_score": -10.0},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.isnan(isos["C"].scores["fep"])
        assert np.isnan(isos["CC"].scores["fep"])
        assert np.isnan(isos["CCC"].scores["fep"])
        assert np.isnan(isos["CCCO"].scores["fep"])

    def test_MakeAbsolute_subgraph(
        self,
        tmp_path: Path,
        test_config: Config,
        result_network_sub: tuple[list[IsomerCollection], dict[tuple[str, str], FEPResult]],
    ) -> None:
        ref = Isomer.from_smiles("CC")
        mols, data = result_network_sub
        rig = TestRig(MakeAbsolute, config=test_config)
        res = rig.setup_run(
            inputs={"inp": [data], "inp_mols": [mols], "inp_ref": [ref]},
            parameters={"ref_score": -10.0},
        )
        new_mols = res["out"].get()
        assert new_mols is not None
        isos = {iso.to_smiles(remove_h=True): iso for mol in new_mols for iso in mol.molecules}
        assert np.isnan(isos["C"].scores["fep"])
        assert np.allclose(isos["CC"].scores["fep"], -10.0, 0.1)
        assert np.allclose(isos["CCC"].scores["fep"], -9.01, 0.1)
        assert np.allclose(isos["CCCO"].scores["fep"], -8.69, 0.1)

    def test_SaveOpenFEResults(
        self,
        tmp_path: Path,
        test_config: Config,
        ligand_path: Path,
        ref_path: Path,
    ) -> None:
        ref = Isomer.from_sdf(ref_path)
        mol = Isomer.from_sdf(ligand_path)
        data = {
            (ref.inchi, mol.inchi): FEPResult(
                smiles=(ref.to_smiles(), mol.to_smiles()),
                ddg=-1.0,
                ddg_error=0.5,
                mapping_score=0.5,
            )
        }
        path = tmp_path / "out.csv"
        rig = TestRig(SaveOpenFEResults, config=test_config)
        rig.setup_run(
            inputs={"inp": [data]},
            parameters={"file": path},
        )

        assert path.exists()
        with path.open("r") as inp:
            reader = csv.reader(inp)
            row = next(reader)
            assert row == ["origin", "target", "origin-smiles", "target-smiles", "ddg", "error"]
            row = next(reader)
            assert row[0].startswith("ILBZVTXOJSVUIM")
            assert row[1].startswith("UZYTVPMNMQLENF")
            assert float(row[4]) == -1.0
            assert float(row[5]) == 0.5
