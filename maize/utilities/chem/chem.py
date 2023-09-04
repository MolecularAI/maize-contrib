"""Utility functions and types for chemistry."""

from collections import defaultdict
from contextlib import redirect_stderr
import io
import logging
from pathlib import Path
import signal
from types import FrameType
from typing import Any, Callable, List, Literal, NoReturn, Sequence, TypeVar, ParamSpec, cast
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdFMCS, rdchem, rdGeometry
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

from maize.utilities.execution import CommandRunner
from maize.utilities.validation import FailValidator

rdBase.WrapLogs()

log = logging.getLogger("run")

# This is vital to have all properties transferred properly between processes
# See also: https://github.com/rdkit/rdkit/issues/1320
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

# Allows partial sanitization of charged molecules with non-standard valences
_SANITIZE_FLAGS = (
    Chem.SanitizeFlags.SANITIZE_FINDRADICALS
    | Chem.SanitizeFlags.SANITIZE_KEKULIZE
    | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
    | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
    | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
    | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
)
SCHRODINGER_VARIANT_TAG = "s_lp_Variant"


class ChemistryException(Exception):
    """Exception raised for issues with executing RDKit or Openbabel"""


_P = ParamSpec("_P")
_T = TypeVar("_T")


def result_check(
    func: Callable[_P, _T], none_check: bool = True, false_check: bool = False
) -> Callable[_P, _T]:
    """
    Calls an RDKit or Openbabel function and raises an exception if
    something goes wrong, instead of just returning ``None``.

    Parameters
    ----------
    func
        The function to wrap
    none_check
        Whether to check for a ``None`` return value
    false_check
        Whether to check for a ``False`` return value

    Returns
    -------
    Callable[_P, _T]
        The wrapped function

    """

    # RDKit does not throw exceptions, but just returns `None`
    # if something went wrong. It also writes the cause to its
    # internal logger, so we need to redirect this output to
    # stderr (with the call to `WrapLogs()` above) and then
    # capture it to get useful information. The same goes for
    # Openbabel, except that it returns ``False`` on failure.
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        logger = io.StringIO()
        with redirect_stderr(logger):
            res = func(*args, **kwargs)

        out = logger.getvalue()
        if (
            (none_check and res is None)
            or (false_check and not res)
            or any(val in out for val in ("error", "Failed"))
        ):
            raise ChemistryException(
                f"{func.__name__} raised the following error: {logger.getvalue()}"
            )
        return res

    return wrapped


def _timeout(timeout: int) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Allows a wrapped function to timeout by raising a `TimeoutError`.

    Parameters
    ----------
    timeout
        Timeout in seconds, has to be an integer value

    Returns
    -------
    Callable[[Callable[_P, _T]], Callable[_P, _T]]
        Wrapped function

    Raises
    ------
    TimeoutError
        If the wrapped function times out

    """

    def _handler(signum: int, frame: FrameType | None) -> NoReturn:
        raise TimeoutError("Function timed out")

    def wrapper(func: Callable[_P, _T]) -> Callable[_P, _T]:
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            signal.signal(signal.SIGALRM, handler=_handler)
            signal.alarm(timeout)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)

        return inner

    return wrapper


def convert(files: list[Path] | Path, output: Path, extra: list[str] | None = None) -> None:
    """
    Convert one or more molecule files into another using openbabel.
    Will detect the input and output formats from the suffixes.

    Parameters
    ----------
    file
        Input file(s) to convert, if multiple are given will
        attempt to aggregate output into a single file
    output
        Output file path
    extra
        Additional arguments to pass to ``obabel``

    """
    if isinstance(files, Path):
        files = [files]

    inp_format = files[0].suffix.strip(".")
    out_format = output.suffix.strip(".")
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"File at '{file.as_posix()}' not found")

    all_files = " ".join(file.as_posix() for file in files)
    command = f"obabel {all_files} -i{inp_format} -o{out_format} -O {output.as_posix()}"
    extra = extra or []
    for arg in extra:
        command += " " + arg

    cmd = CommandRunner(validators=[FailValidator("error")])
    cmd.run_validate(command, verbose=True)


def save_sdf_library(
    file: Path,
    mols: list["IsomerCollection"],
    conformers: bool = False,
    split_strategy: Literal["schrodinger", "inchi"] = "inchi",
) -> None:
    """
    Saves a list of IsomerCollection instances to an SDF file.
    Will only write one Isomer and one conformer for each molecule.

    Parameters
    ----------
    file
        Path to the output SDF file
    smiles
        List of `IsomerCollection` to write
    conformers
        Whether to write all conformers
    split_strategy
        How to determine which isomer belongs to which molecule, from special tags or
        the name. ``schrodinger`` expects a name like ``1:0`` (molecule:isomer), ``inchi``
        expects an InChI key as a name, which can be split into molecule and isomer.

    """
    with Chem.SDWriter(file.as_posix()) as writer:
        for i, mol in enumerate(mols):
            for j, iso in enumerate(mol.molecules):
                if split_strategy == "schrodinger":
                    iso.name = f"{i}:{j}"
                elif split_strategy == "inchi":
                    iso.name = iso.inchi

                if conformers:
                    for i in range(iso.n_conformers):
                        writer.write(iso._molecule, confId=i)
                else:
                    writer.write(iso._molecule)


def load_sdf_library(
    file: Path, split_strategy: Literal["schrodinger", "inchi", "schrodinger-tag", "none"] = "inchi"
) -> list["IsomerCollection"]:
    """
    Loads an SDF library containing multiple molecules with potentially
    multiple isomers each, and creates an `IsomerCollection` for all of them.

    Parameters
    ----------
    file
        Path to the SDF file
    split_strategy
        How to determine which isomer belongs to which molecule, from special tags or
        the name. ``schrodinger`` expects a name like ``1:0`` (molecule:isomer),
        ``schrodinger-tag`` expects a tag named ``s_lp_Variant``, ``inchi`` expects an
        InChI key as a name, which can be split into molecule and isomer, and ``'none'``
        provides a separate `IsomerCollection` for each entry.

    Returns
    -------
    list[IsomerCollection]
        Created molecules

    """
    mols: dict[int | str, dict[int | str, list[Chem.rdchem.Mol]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with Chem.SDMolSupplier(file.as_posix(), removeHs=False) as supp:
        for i, mol in enumerate(supp):
            if mol is None:
                continue
            iso = Isomer(mol)

            mol_idx: int | str
            # Split based on special name: molecule:isomer
            if split_strategy == "schrodinger":
                if iso.name is None or len(parts := iso.name.split(":")) < 2:
                    raise ChemistryException(
                        f"Parsed molecule {iso} does not have a suitable name,"
                        "expected format molecule-index:isomer-index"
                    )
                mol_idx, iso_idx = (int(part) for part in parts)
                mols[mol_idx][iso_idx].append(mol)

            # Split based on special tag created by Schrodinger
            elif split_strategy == "schrodinger-tag":
                if not iso.has_tag(SCHRODINGER_VARIANT_TAG):
                    raise ChemistryException(
                        f"Parsed molecule does not have the expected '{SCHRODINGER_VARIANT_TAG}'"
                    )

                # The tag will be something like 'file.smi:1-1' with the indices
                # representing molecule and isomer number, respectively
                tag = iso.get_tag(SCHRODINGER_VARIANT_TAG)
                assert isinstance(tag, str)
                if ":" in tag:
                    mol_idx, iso_idx = (int(part) for part in tag.split(":")[-1].split("-"))
                else:
                    mol_idx, iso_nr = tag.split("-")
                    iso_idx = int(iso_nr)
                mols[mol_idx][iso_idx].append(mol)

            # Split based on InChI key: moleculeinchi-isomerinchi (e.g. ABCDEF-GHIJKL-M)
            elif split_strategy == "inchi":
                if iso.name is None or len(inchi_parts := iso.name.split("-")) != 3:
                    raise ChemistryException(
                        f"Name of parsed molecule ('{iso.name}') is not a valid InChI key"
                    )
                mol_part, *iso_part = inchi_parts
                mols[mol_part]["-".join(iso_part)].append(mol)

            # No split, each entry is a separate molecule
            elif split_strategy == "none":
                mols[i][0].append(mol)

    return [
        IsomerCollection([Isomer.from_rdmols(isos) for isos in mol.values()])
        for mol in mols.values()
    ]


def _lib2dict(lib: list["IsomerCollection"]) -> dict[str, dict[str, "Isomer"]]:
    """Provides a nested dictionary from a library using InChI keys"""
    dic: dict[str, dict[str, "Isomer"]] = defaultdict(dict)
    for mol in lib:
        for iso in mol.molecules:
            mol_id, iso_id, state = iso.inchi.split("-")
            dic[mol_id][f"{iso_id}-{state}"] = iso
    return dic


def _dict2lib(dic: dict[str, dict[str, "Isomer"]]) -> list["IsomerCollection"]:
    """Provides a list of `IsomerCollection` from a nested dictionary"""
    return [IsomerCollection(list(isodict.values())) for isodict in dic.values()]


def merge_libraries(
    base: list["IsomerCollection"], other: list["IsomerCollection"]
) -> list["IsomerCollection"]:
    base_dic, other_dic = _lib2dict(base), _lib2dict(other)
    for key, iso_dic in base_dic.items():
        if key in other_dic:
            iso_dic.update(other_dic[key])
    return _dict2lib(base_dic)


def save_smiles(file: Path, smiles: list[str]) -> None:
    """
    Saves a list of SMILES to a ``.smi`` file.

    Parameters
    ----------
    file
        Path to the output file
    smiles
        List of SMILES codes to write

    """
    if file.suffix != ".smi":
        raise ValueError("File needs to have the .smi suffix")

    with open(file, "w", encoding="utf8") as out:
        out.writelines("\n".join(smiles))


def mcs(*mols: "Isomer") -> "Isomer":
    """
    Finds the maximum common substructure (MCS) between multiple molecules.

    Parameters
    ----------
    mols
        Reference `Isomer` molecules

    Returns
    -------
    Isomer
        Isomer instance using the MCS

    """
    mcs = rdFMCS.FindMCS([mol._molecule for mol in mols])
    return Isomer(result_check(Chem.MolFromSmarts)(mcs.smartsString))


def rmsd(mol: "Isomer", ref: "Isomer") -> NDArray[np.float32]:
    """
    Calculate the unaligned root-mean-square deviation between
    all conformers of an isomer and a reference molecule.

    Parameters
    ----------
    mol
        Molecule in the form of an `Isomer`
    ref
        Reference molecule

    Returns
    -------
    NDArray[np.float32]
        RMSDs for all conformers

    """
    inds_ref, inds_mol = np.array(mol.atommap(ref)).T
    diff = mol.coordinates[:, inds_mol] - ref.coordinates[:, inds_ref]
    return cast(NDArray[np.float32], np.sqrt((diff**2).sum(axis=-1).mean(axis=1)))


_TAG_METHODS = {
    bool: "SetBoolProp",
    int: "SetIntProp",
    float: "SetDoubleProp",
}


_ValidRDKitTagType = bool | int | float | str | list[float | int] | NDArray[np.float_ | np.int_]


def _prop_setter(
    obj: Chem.rdchem.Conformer | Chem.rdchem.Mol | Chem.rdchem.Atom,
    tag: str,
    value: _ValidRDKitTagType,
) -> None:
    """Sets RDKit properties using the appropriate setter."""

    # Arrays require a conversion to a string; numpy arrays and
    # lists serialize differently (with and without commas as
    # delimiters), so we convert to list by default
    if isinstance(value, (list, np.ndarray)):
        value = str(list(value))

    # Numpy floats and ints need special treatment
    elif isinstance(value, (float, np.float_)):
        value = float(value)

    elif isinstance(value, (int, np.int_)):
        value = int(value)

    # Should just be a string
    if not isinstance(value, (bool, int, float, str)):
        value = str(value)

    # `SetProp` is the default for all strings
    getattr(obj, _TAG_METHODS.get(type(value), "SetProp"))(tag, value)


def _prop_converter(raw: _ValidRDKitTagType) -> _ValidRDKitTagType:
    """Convert raw RDKit tags to an appropriate python type"""
    if isinstance(raw, (bool, int, float)):
        return raw

    # This should never happen
    if not isinstance(raw, str):
        raise TypeError(f"Got invalid type '{type(raw)}' for value '{raw}' from rdkit tag")

    # We have an array and attempt to deserialize to an ndarray of floats
    if raw.startswith("[") and raw.endswith("]"):
        return np.fromstring(raw[1:-1], sep=",")
    return raw


class Conformer:
    """
    Thin shim layer for rdkit conformers. Each conformer will have an `Isomer` as its parent.

    Parameters
    ----------
    rd_conf
        RDKit conformer object
    parent
        `Isomer` parent instance
    _rd_parent
        The actual rdkit molecule parent of the conformer

    """

    def __init__(
        self,
        rd_conf: Chem.rdchem.Conformer,
        parent: "Isomer",
        _rd_parent: Chem.rdchem.Mol | None = None,
    ) -> None:
        self._conf = rdchem.Conformer(rd_conf)
        self.parent = parent

        # We need this to ensure the parent rdkit molecule instance does not go out of scope,
        # as this means we would lose access to our conformer (conformers cannot exist alone
        # and always need a parent molecule instance to exist). See this related issue for a
        # (now fixed) example: https://github.com/rdkit/rdkit/issues/3492
        self._rd_parent = parent._molecule if _rd_parent is None else _rd_parent

        coords = []
        for i in range(self._conf.GetNumAtoms()):
            pos = self._conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
        self._coordinates = np.array(coords)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_atoms={self.n_atoms}, "
            f"orphan={not self._conf.HasOwningMol()})"
        )

    @classmethod
    def from_rdmol(cls, rd_mol: Chem.rdchem.Mol, parent: "Isomer", renumber: bool = True) -> Self:
        """
        Create a conformer from an RDKit molecule.

        Parameters
        ----------
        rd_mol
            RDKit molecule instance
        parent
            Parent `Isomer` instance
        renumber
            Ensure atom numbers of the molecule match the parent

        """
        if (n_atoms_conf := rd_mol.GetNumAtoms()) != parent.n_atoms:
            raise ChemistryException(
                f"Atom number mismatch, parent molecule has {parent.n_atoms}, "
                f"conformer has {n_atoms_conf}"
            )

        # We can't fully sanitize as we might have charged molecules
        # (for example with N valence of 4), so we disable sanitization
        # at load time and do a partial sanitizing step after, see also:
        # https://www.rdkit.org/docs/Cookbook.html#explicit-valence-error-partial-sanitization
        Chem.SanitizeMol(rd_mol, _SANITIZE_FLAGS, catchErrors=True)

        # We ensure that the atom numbering is consistent between the new conformer and the parent,
        # as the passed in molecule might not have any relation to the parent. This can be important
        # when converting from files that treat protonations differently, for example PDBQT.
        # Based on: https://www.rdkit.org/docs/Cookbook.html#reorder-atoms
        if renumber:
            parent_canon_order = np.array(Chem.CanonicalRankAtoms(parent._molecule)).argsort()
            newmol_canon_order = np.array(Chem.CanonicalRankAtoms(rd_mol)).argsort()
            inds = np.arange(parent.n_atoms)
            inds[parent_canon_order] = newmol_canon_order

            # FIXME This fails occasionally, example:
            # C1[C@H]2[C@@H]([C@@H](S1)CCCCC(=O)O)NC(=S)N2
            rd_mol = Chem.RenumberAtoms(rd_mol, [int(i) for i in inds])

        rd_conf = result_check(rd_mol.GetConformer)(0)
        conf = cls(rd_conf=rd_conf, parent=parent, _rd_parent=rd_mol)

        # Properties are not automatically copied over to the conformer
        for key, value in rd_mol.GetPropsAsDict().items():
            conf.set_tag(key, value)

        return conf

    @property
    def n_atoms(self) -> int:
        return cast(int, result_check(self._conf.GetNumAtoms)())

    @property
    def tags(self) -> dict[str, Any]:
        """Return all tags"""
        return cast(dict[str, Any], self._conf.GetPropsAsDict())

    @property
    def coordinates(self) -> NDArray[np.float32]:
        """Coordinates of the conformer"""
        if self._conf.HasOwningMol():
            coords = []
            for i in range(self._conf.GetNumAtoms()):
                pos = self._conf.GetAtomPosition(i)
                coords.append([pos.x, pos.y, pos.z])
            return np.array(coords)
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coords: NDArray[np.float32]) -> None:
        """Set the coordinates"""
        if self._conf.HasOwningMol():
            self._set_conf_coords(coords)
        self._coordinates = coords

    # RDKit conformers are intrinsically tied to their parent molecule instance.
    # Therefore, calling `RemoveConformer` on the parent will partially deallocate
    # the memory associated with the conformer, leaving an orphan instance with no
    # atoms or coordinates. We thus need to internally keep track of the coordinates
    # and regenerate the Conformer wrapper manually when assigning an existing
    # conformer to an RDKit molecule, regardless of if it's been orphaned or not.
    # See this classic RDKit bug: https://github.com/rdkit/rdkit/issues/3817
    def regenerate(self) -> None:
        """Recreate the internal RDKit conformer object to avoid deallocation"""
        self._set_conf_coords(coords=self._coordinates)

    def set_tag(self, tag: str, value: _ValidRDKitTagType) -> None:
        """
        Sets a tag / property for the whole isomer.

        Parameters
        ----------
        tag
            Tag to set
        value
            Corresponding value for all conformers

        """
        if not self._conf.HasOwningMol():
            log.warning("Conformer is orphaned and will not remember any set tags")
        _prop_setter(self._conf, tag, value)

    def get_tag(self, tag: str, default: _ValidRDKitTagType | None = None) -> _ValidRDKitTagType:
        """
        Returns the value for the specified tag.

        Parameters
        ----------
        tag
            The tag to lookup
        default
            A default value to return if the key is not found

        Returns
        -------
        Any
            The value of the tag

        Raises
        ------
        KeyError
            If the specified tag couldn't be found

        """
        if tag not in self.tags:
            if default is None:
                raise KeyError(f"Tag '{tag}' not found in molecule")
            return default
        return _prop_converter(self.tags[tag])

    def has_tag(self, tag: str) -> bool:
        """
        Returns whether the specified tag is set.

        Parameters
        ----------
        tag
            The tag to lookup

        Returns
        -------
        bool
            Whether the tag is defined

        """
        return cast(bool, result_check(self._conf.HasProp)(tag))

    def _set_conf_coords(self, coords: NDArray[np.float32]) -> None:
        """Set the coordinates of the wrapped RDKit conformer"""
        for i in range(self._conf.GetNumAtoms()):
            self._conf.SetAtomPosition(i, rdGeometry.Point3D(*coords[i]))


class Isomer:
    """
    Thin shim layer for rdkit molecules. Here, an isomer refers
    to a unique chemical form of a molecule, i.e. a form separated
    by major energy barriers. Note that SMILES codes do not
    necessarily map to a single isomer! Some examples of unique isomers:

    * A ring conformer (e.g. for cyclohexane)
    * Cis-trans isomers
    * Chirality
    * Tautomers
    * Different protonation states

    Parameters
    ----------
    rd_mol
        RDKit molecule instance

    """

    def __init__(self, rd_mol: Chem.rdchem.Mol) -> None:
        self._molecule = rd_mol
        self._init_conformers()
        self.score_tag = "score"

    def __repr__(self) -> str:
        scores = f", scores={self.scores}" if self.scored else ""
        return (
            f"{self.__class__.__name__}(n_atoms={self.n_atoms}, "
            f"n_conformers={self.n_conformers}, charge={self.charge}{scores})"
        )

    # RDKit is for some reason unable to pickle conformer objects directly,
    # but only as part of an rdmol object. Because `Isomer` keeps references
    # to our Conformer wrapper objects, which in turn have a reference to the
    # RDKit conformer, pickling will fail. We therefore remove our conformer
    # proxies before pickling and restore them later with `_init_conformers()`.
    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
        self._init_conformers()

    def __getstate__(self) -> dict[str, Any]:
        inst = self.__dict__.copy()
        inst["_conformers"] = []
        return inst

    @classmethod
    def from_rdmols(cls, rd_mols: list[Chem.rdchem.Mol], sanitize: bool = True) -> Self:
        """
        Create a molecule from multiple RDKit molecules acting as conformers.

        Parameters
        ----------
        rd_mols
            List of RDKit molecule instances
        sanitize
            Whether to sanitize the molecule

        Returns
        -------
        Molecule
            Molecule instance

        """
        first, *rest = rd_mols
        iso = cls(first)
        for conf in rest:
            iso.add_conformer(Conformer.from_rdmol(conf, parent=iso))
        return iso

    @classmethod
    def from_smiles(cls, smiles: str, sanitize: bool = True) -> Self:
        """
        Create a molecule from a SMILES string.

        Parameters
        ----------
        smiles
            The SMILES string to initialize the molecule with
        sanitize
            Whether to sanitize the molecule

        Returns
        -------
        Molecule
            Molecule instance

        Raises
        ------
        ChemistryException
            If there was an error parsing the SMILES code

        """
        rd_mol = result_check(Chem.MolFromSmiles)(smiles, sanitize=sanitize)
        return cls(rd_mol=rd_mol)

    @classmethod
    def from_sdf(cls, file: Path, read_conformers: bool = False) -> Self:
        """
        Create a molecule from an SDF file.

        Parameters
        ----------
        file
            The SDF file to initialize the molecule with
        read_conformers
            Whether to read all conformers

        Returns
        -------
        Molecule
            Molecule instance

        Raises
        ------
        ChemistryException
            If there was an error parsing the SDF

        """
        rd_mols = []
        with Chem.SDMolSupplier(file.as_posix()) as suppl:
            for rd_mol in suppl:
                if rd_mol is None:
                    continue
                if not read_conformers:
                    return cls(rd_mol)
                rd_mols.append(rd_mol)
        if not rd_mols:
            raise ChemistryException(f"SDF file '{file.name}' produced an empty molecule")

        return cls.from_rdmols(rd_mols)

    @classmethod
    def from_sdf_block(cls, sdf: str) -> Self:
        """
        Create a molecule from an SDF string.

        Parameters
        ----------
        sdf
            The SDF string to initialize the molecule with

        Returns
        -------
        Molecule
            Molecule instance

        Raises
        ------
        ChemistryException
            If there was an error parsing the SDF

        """
        rd_mol = result_check(Chem.MolFromMolBlock)(sdf)
        return cls(rd_mol=rd_mol)

    @property
    def name(self) -> str | None:
        """Returns the molecule name"""
        if self._molecule.HasProp("_Name"):
            return cast(str, self._molecule.GetProp("_Name"))
        return None

    @name.setter
    def name(self, value: str) -> None:
        self._molecule.SetProp("_Name", value)

    @property
    def inchi(self) -> str:
        """Returns the InChI key for the molecule"""
        # We need the fixed-H option here to ensure unique InChIs for different
        # protonation states, see: https://www.inchi-trust.org/technical-faq-2/#15.24
        return cast(str, result_check(Chem.MolToInchiKey)(self._molecule, options="-FixedH"))

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule"""
        return cast(int, self._molecule.GetNumAtoms())

    @property
    def n_conformers(self) -> int:
        """Number of conformers in the molecule"""
        return cast(int, self._molecule.GetNumConformers())

    @property
    def charge(self) -> int:
        """The charge of the molecule"""
        return cast(int, result_check(Chem.GetFormalCharge)(self._molecule))

    @property
    def scores(self) -> NDArray[np.float32]:
        """The scores of the molecule"""
        conf_scores = np.array(
            [conf.get_tag(self.score_tag, np.nan) for conf in self.conformers]
        ).flatten()
        iso_scores = np.array(self.get_tag(self.score_tag, [np.nan])).flatten()
        if (
            conf_scores is None
            or len(conf_scores) < len(iso_scores)
            or (~np.isfinite(conf_scores)).all()
        ):
            return iso_scores
        return conf_scores

    @scores.setter
    def scores(self, scores: NDArray[np.float32]) -> None:
        """Set the scores"""
        self.set_tag(self.score_tag, scores)
        for conf, score in zip(self.conformers, scores):
            conf.set_tag(self.score_tag, score)

    @property
    def scored(self) -> bool:
        """Whether the molecule was scored"""
        return len(self.scores) > 0 and bool(np.isfinite(self.scores).any())

    @property
    def conformers(self) -> list[Conformer]:
        """Get all stored isomer conformers in an iterable structure"""
        return self._conformers

    @property
    def coordinates(self) -> NDArray[np.float32]:
        """Get coordinates of all stored conformers as one NDArray"""
        return np.array([conf.coordinates for conf in self.conformers])

    @property
    def tags(self) -> dict[str, Any]:
        """Return all tags"""
        return cast(dict[str, Any], self._molecule.GetPropsAsDict())

    def to_sdf(self, path: Path, write_conformers: bool = False) -> None:
        """
        Generate an SDF file for the isomer.

        Parameters
        ----------
        path
            Output file path
        write_conformers
            Whether to write all conformers

        """
        with Chem.SDWriter(path.as_posix()) as writer:
            if write_conformers:
                for conf in self._molecule.GetConformers():
                    writer.write(self._molecule, confId=conf.GetId())
            else:
                writer.write(self._molecule)

    def to_smiles(self) -> str:
        """Generate a SMILES code for the isomer"""
        return cast(str, result_check(Chem.MolToSmiles)(self._molecule))

    def to_mol_block(self) -> str:
        """Generate a MOL block for the isomer."""
        return cast(str, Chem.MolToMolBlock(self._molecule))

    def addh(self) -> None:
        """Adds hydrogens to the molecule"""
        self._molecule = Chem.AddHs(self._molecule)

    def set_tag(self, tag: str, value: _ValidRDKitTagType) -> None:
        """
        Sets a tag / property for the whole isomer.

        Parameters
        ----------
        tag
            Tag to set
        value
            Corresponding value for all conformers

        """
        _prop_setter(self._molecule, tag, value)

    def get_tag(self, tag: str, default: _ValidRDKitTagType | None = None) -> _ValidRDKitTagType:
        """
        Returns the value for the specified tag.

        Parameters
        ----------
        tag
            The tag to lookup
        default
            A default value to return if the key is not found

        Returns
        -------
        Any
            The value of the tag

        Raises
        ------
        KeyError
            If the specified tag couldn't be found

        """
        if tag not in self.tags:
            if default is None:
                raise KeyError(f"Tag '{tag}' not found in molecule")
            return default
        return _prop_converter(self.tags[tag])

    def has_tag(self, tag: str) -> bool:
        """
        Returns whether the specified tag is set.

        Parameters
        ----------
        tag
            The tag to lookup

        Returns
        -------
        bool
            Whether the tag is defined

        """
        return cast(bool, result_check(self._molecule.HasProp)(tag))

    def set_atomic_tag(self, idx: int, tag: str, value: _ValidRDKitTagType) -> None:
        """
        Sets a tag / property for a single atom.

        Parameters
        ----------
        idx
            Index of the atom
        tag
            Tag to set
        value
            Corresponding value for the atom

        """
        atom = result_check(self._molecule.GetAtomWithIdx)(idx)
        _prop_setter(atom, tag, value)

    def get_atomic_tag(self, idx: int, tag: str, default: Any = None) -> Any:
        """
        Returns the value for the specified tag of an atom.

        Parameters
        ----------
        idx
            Index of the atom
        tag
            The tag to lookup
        default
            A default value to return if the key is not found

        Returns
        -------
        Any
            The value of the tag

        Raises
        ------
        KeyError
            If the specified tag couldn't be found

        """
        atom = result_check(self._molecule.GetAtomWithIdx)(idx)
        if not atom.HasProp(tag):
            if default is None:
                raise KeyError(f"Tag '{tag}' not found in atom {idx}")
            return default
        return atom.GetProp(tag)

    def has_atomic_tag(self, idx: int, tag: str) -> bool:
        """
        Returns whether the specified atomic tag is set.

        Parameters
        ----------
        idx
            Index of the atom
        tag
            The tag to lookup

        Returns
        -------
        bool
            Whether the tag is defined

        """
        atom = result_check(self._molecule.GetAtomWithIdx)(idx)
        return bool(result_check(atom.HasProp)(tag))

    def add_conformer(self, conf: Conformer) -> None:
        """
        Adds a conformer to the isomer.

        Parameters
        ----------
        conf
            Conformer instance

        """
        conf.regenerate()
        result_check(self._molecule.AddConformer, none_check=False)(conf._conf)
        self._conformers.append(conf)

    def remove_conformer(self, idx: int) -> None:
        """
        Remove a conformer.

        Parameters
        ----------
        idx
            Index of the conformer to be removed

        """
        result_check(self._molecule.RemoveConformer, none_check=False)(idx)
        self._conformers.pop(idx)

    def clear_conformers(self) -> None:
        """Remove all conformers."""
        self._molecule.RemoveAllConformers()
        self._conformers: list[Conformer] = []

    def update_conformers_from_mol_block(
        self, block: str, score_parser: Callable[[dict[str, str]], float] | None = None
    ) -> None:
        """
        Update molecule conformers from a mol block.

        Parameters
        ----------
        mol
            The mol block
        score_parser
            Function used to parse a score from the mol properties

        Raises
        ------
        ChemistryException
            If there was an error parsing the mol block

        """
        self.clear_conformers()
        mol = result_check(Chem.MolFromMolBlock)(block, removeHs=False)
        try:
            conf = Conformer.from_rdmol(mol, parent=self)
            self.add_conformer(conf)
        except ValueError as err:
            raise ChemistryException("Unable to parse conformer, error: %s", err)

        # Only set scores if we actually got some, otherwise they might be set externally
        if score_parser is not None:
            self.scores = np.array(score_parser(mol.GetPropsAsDict()))

    def update_conformers_from_sdf(
        self, sdf: Path, score_parser: Callable[[dict[str, str]], float] | None = None
    ) -> None:
        """
        Update molecule conformers from an SDF file.

        Parameters
        ----------
        sdf
            The SDF file to initialize the molecule with
        score_parser
            Function used to parse a score from the SDF properties

        Raises
        ------
        ChemistryException
            If there was an error parsing the SDF

        """
        scores = []
        self.clear_conformers()
        with Chem.SDMolSupplier(sdf.as_posix(), removeHs=False, sanitize=False) as supp:
            for mol in supp:
                if mol is None:
                    continue
                try:
                    conf = Conformer.from_rdmol(mol, parent=self)
                    self.add_conformer(conf)
                except ValueError as err:
                    log.warning("Unable to parse conformer, error: %s", err)
                    continue

                if score_parser is not None:
                    score = score_parser(mol.GetPropsAsDict())
                    scores.append(score)

        # Only set scores if we actually got some, otherwise they might be set externally
        if score_parser is not None:
            self.scores = np.array(scores)

    def atommap(self, mol: "Isomer") -> list[tuple[int, int]]:
        """
        Finds the atom index mappings based on the MCS.

        Parameters
        ----------
        mol
            Reference `Isomer`

        Returns
        -------
        list[tuple[int, int]]
            Pairs of atom indices

        """
        mcs_mol = mcs(self, mol)
        ref_substructure = result_check(mol._molecule.GetSubstructMatch)(mcs_mol._molecule)
        iso_substructure = result_check(self._molecule.GetSubstructMatch)(mcs_mol._molecule)
        return list(zip(ref_substructure, iso_substructure))

    def generate_stereoisomers(self, n_max: int = 32) -> list[Self]:
        """
        Generates possible enantiomers.

        Parameters
        ----------
        n_max
            Maximum number of stereoisomers to generate

        Returns
        -------
        list[Molecule]
            List of new molecules representing distinct stereoisomers

        """
        if n_max <= 1:
            return [self]
        enum_options = StereoEnumerationOptions(unique=True, maxIsomers=n_max, tryEmbedding=True)
        return [
            self.__class__(isomer)
            for isomer in EnumerateStereoisomers(self._molecule, options=enum_options)
        ]

    def to_pdb(self, path: Path) -> None:
        """
        Writes a molecule to a PDB file.

        Parameters
        ----------
        path
            Path to the PDB file to write

        """
        result_check(Chem.MolToPDBFile, none_check=False)(self._molecule, filename=path.as_posix())
        if not path.exists():
            raise ChemistryException(f"File at '{path.as_posix()}' was not written")

    def embed(self, n_conformers: int = 1) -> None:
        """
        Generate a 3D embedding of the molecule.

        Parameters
        ----------
        n_conformers
            Number of conformers to generate

        Raises
        ------
        ChemistryException
            If there was an error generating the embeddings

        """
        self.clear_conformers()
        self.addh()
        result_check(AllChem.EmbedMultipleConfs)(
            self._molecule, numConfs=n_conformers, maxAttempts=100
        )
        self._init_conformers()

    def _init_conformers(self) -> None:
        """Initializes `Conformer` instances based on contained RDKit conformers."""
        self._conformers = [
            Conformer(rd_conf, parent=self)
            for rd_conf in result_check(self._molecule.GetConformers)()
        ]


class IsomerCollection:
    """
    Represents a collection of isomers / enumerations of a molecule.

    Parameters
    ----------
    molecules
        Molecules part of the collection and sharing a common topology

    """

    smiles: str | None = None

    def __init__(self, molecules: Sequence[Isomer]) -> None:
        self.molecules = list(molecules)

    def __repr__(self) -> str:
        scores = f", best_score={self.best_score:4.4}" if self.scored else ""
        smiles = "" if self.smiles is None else f"'{self.smiles}', "
        return f"{self.__class__.__name__}({smiles}n_isomers={self.n_isomers}{scores})"

    @classmethod
    def from_smiles(
        cls, smiles: str, max_isomers: int = 8, sanitize: bool = True, timeout: int = 10
    ) -> Self:
        """
        Create an isomer collection from a SMILES string.

        Parameters
        ----------
        smiles
            The SMILES string to initialize the collection with
        max_isomers
            Maximum number of isomers to generate
        sanitize
            Whether to sanitize the molecule
        timeout
            Timeout in seconds before failing stereoisomer generation

        Returns
        -------
        IsomerCollection
            IsomerCollection instance

        Raises
        ------
        ChemistryException
            If there was an error parsing the SMILES code

        See Also
        --------
        steps.mai.molecule.Gypsum
            A more advanced approach of generating any
            kind of isomer or high-energy conformers

        """
        mol = Isomer.from_smiles(smiles, sanitize=sanitize)
        try:
            isomers = _timeout(timeout)(mol.generate_stereoisomers)(n_max=max_isomers)
        except TimeoutError:
            log.warning(f"Stereoisomer generation for {mol} (SMILES={smiles}) timed out")
            isomers = []
        collection = cls(isomers)
        collection.smiles = smiles
        return collection

    @classmethod
    def from_sdf(cls, file: Path) -> Self:
        """
        Create an isomer collection from an SDF file
        containing different isomers or conformers.

        Parameters
        ----------
        file
            The SDF file input

        Returns
        -------
        IsomerCollection
            IsomerCollection instance

        Raises
        ------
        ChemistryException
            If there was an error parsing the SMILES code
        """
        mols = []
        smiles = None
        with Chem.SDMolSupplier(file.as_posix(), removeHs=False) as supp:
            for mol in supp:
                if mol is None:
                    continue
                mols.append(Isomer(mol))
                if mol.HasProp("SMILES"):
                    smiles = mol.GetProp("SMILES")
        coll = cls(mols)
        if smiles is not None:
            coll.smiles = smiles
        return coll

    @property
    def n_isomers(self) -> int:
        """Number of contained isomers"""
        return len(self.molecules)

    @property
    def scored(self) -> bool:
        """Whether the isomers have been scored"""
        return all(iso.scored for iso in self.molecules)

    @property
    def best_score(self) -> float:
        """The best score among all isomers / conformers"""
        if self.scored and self.n_isomers > 0:
            it = (cast(float, mol.scores.min()) for mol in self.molecules if mol.scores is not None)
            return min(it)
        return np.nan

    @best_score.setter
    def best_score(self, score: float) -> None:
        for mol in self.molecules:
            mol.scores = np.array([score])

    @property
    def inchi(self) -> str:
        """Returns the InChI key for the molecule"""
        if self.n_isomers > 0:
            return self.molecules[0].inchi.split("-")[0]
        return ""

    def to_sdf(self, path: Path) -> None:
        """
        Write all isomers to an SDF file.

        Parameters
        ----------
        path
            Output file path
        """
        with Chem.SDWriter(path.as_posix()) as writer:
            for iso in self.molecules:
                writer.write(iso._molecule)
    def to_smiles(self) -> List[str]:
        """
        return smiles for an isomer collection
        """
        return [iso.to_smiles() for iso in self.molecules]

    def embed(self, n_conformers: int = 1) -> None:
        """
        Generate a 3D embedding of all isomers.

        Parameters
        ----------
        n_conformers
            Number of conformers to generate

        Raises
        ------
        ChemistryException
            If there was an error generating the embeddings

        """
        for mol in self.molecules:
            mol.embed(n_conformers)

    def remove_isomer(self, isomer: Isomer) -> None:
        """Remove isomer from collection"""
        self.molecules.remove(isomer)
