"""Nodes for sorting isomers"""

import operator as op

from maize.core.node import Node
from maize.core.interface import Input, MultiOutput, Parameter
from maize.utilities.chem import IsomerCollection, Isomer
from maize.utilities.chem.chem import ValidRDKitTagType
from maize.utilities.testing import TestRig
from maize.utilities.io import Config


class TagSorter(Node):
    """
    Sorts molecules to different outputs based on tag values.

    Connect as many nodes to the output as desired, and supply an identical number
    of predicates. These are simple strings including the tag name, a comparison
    operator, and a value to compare against. You can also chain multiple comparisons
    using ``'&'`` (and) or ``'|'`` (or). If a molecule matches the first predicate,
    it will be sent to the first output, if it matches the second predicate, it will
    be sent to the second, and so on.  If a molecule matches multiple predicates it
    will be sent to multiple outputs.

    """
    tags = {"sorter", "chemistry"}

    OPS = {
        ">": op.gt,
        ">=": op.ge,
        "<": op.lt,
        "<=": op.le,
        "==": op.eq,
        "!=": op.ne,
        "&": op.and_,
        "|": op.or_,
    }

    inp: Input[list[IsomerCollection]] = Input()
    """List of molecules as isomer collections"""

    out: MultiOutput[list[IsomerCollection]] = MultiOutput()
    """List of molecules as isomer collections after sorting"""

    sorter: Parameter[list[str]] = Parameter(default_factory=list)
    """
    Tag sorting predicate strings, specify the name of the tag together
    with a boolean operator and a value. In this example, molecules will
    be sorted to three outputs depending on the value of their score tag:

    .. code-block:: python

       node.sorter.set([
           "score < -9",
           "score < -5 & score > -9",
           "score > -5",
       ])

    Allowed operators are '>', '>=', '<', '<=', '==', '!=',
    allowed conjunctions are '&' and '|'.

    """

    @staticmethod
    def _check(mol: Isomer, tag: str, op: str, val: ValidRDKitTagType) -> bool:
        """Checks a molecule with a specific tag against a predicate"""
        mol_value = mol.get_tag(tag)
        if isinstance(mol_value, int | float):
            val = float(val)  # type: ignore
        return bool(TagSorter.OPS[op](mol_value, val))

    # Older versions of mypy will throw an error here, but this can be ignored
    def _matcher(self, mol: Isomer, tokens: list[str]) -> bool:
        """Matches a molecule against a sorting predicate string"""
        match tokens:
            case [tag, op, val] if mol.has_tag(tag):
                return TagSorter._check(mol, tag, op, val)
            case [val, op, tag] if mol.has_tag(tag):
                return TagSorter._check(mol, tag, op, val)
            case [tok_a, op, tok_b, conj, *rest]:
                return bool(
                    TagSorter.OPS[conj](
                        self._matcher(mol, [tok_a, op, tok_b]), self._matcher(mol, rest)
                    )
                )
            case _:
                self.logger.warning(
                    "Invalid predicate '%s' for isomer '%s'", " ".join(tokens), mol.name or mol.inchi
                )
                return False

    def run(self) -> None:
        mols = self.inp.receive()

        for i, (line, out) in enumerate(zip(self.sorter.value, self.out)):
            outs: list[IsomerCollection] = []
            for mol in mols:
                matching_isos = [iso for iso in mol.molecules if self._matcher(iso, line.split())]
                if matching_isos:
                    outs.append(IsomerCollection(matching_isos))

            self.logger.info("Sending %s isomers to output %s (using '%s')", len(outs), i, line)
            out.send(outs)


class TestSuiteSorter:
    def test_TagSorter(self, test_config: Config) -> None:
        isos = [
            Isomer.from_smiles("C"),
            Isomer.from_smiles("C"),
            Isomer.from_smiles("CC"),
            Isomer.from_smiles("CCC"),
        ]

        isos[0].set_tag("score", -11.0)
        isos[1].set_tag("score", -1.0)
        isos[2].set_tag("score", -7.5)
        isos[3].set_tag("score", -3.0)
        mols = [
            IsomerCollection([isos[0], isos[1]]),
            IsomerCollection([isos[2]]),
            IsomerCollection([isos[3]]),
        ]
        sorter = [
            "score < -9",
            "score < -5 & score > -9",
            "score > -5",
        ]
        rig = TestRig(TagSorter, config=test_config)
        result = rig.setup_run_multi(
            inputs={"inp": [mols]}, parameters={"sorter": sorter}, n_outputs=3
        )
        assert isinstance(result["out"], list)
        res = result["out"][0].get()
        assert res is not None
        assert len(res) == 1
        assert res[0].n_isomers == 1
        assert res[0].molecules[0].get_tag("score") == -11.0

        res = result["out"][1].get()
        assert res is not None
        assert len(res) == 1
        assert res[0].n_isomers == 1
        assert res[0].molecules[0].get_tag("score") == -7.5

        res = result["out"][2].get()
        assert res is not None
        assert len(res) == 2
        assert res[0].n_isomers == 1
        assert res[0].molecules[0].get_tag("score") == -1.0
        assert res[1].molecules[0].get_tag("score") == -3.0
