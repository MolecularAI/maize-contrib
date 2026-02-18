"""An example node for maize."""

# pylint: disable=import-outside-toplevel, import-error

# Import any maize modules you might need.
# DO NOT import custom packages here! Put them under `run()`!
import pytest
from maize.core.node import Node
from maize.core.interface import Output, Parameter
from maize.utilities.testing import TestRig


class Example(Node):
    """
    Give your node a brief description here.

    Notes
    -----
    Notes about installing the wrapped tool if applicable.

    References
    ----------
    List your references here!

    """
    # Specify non-standard required commandline tools here,
    # loading will be attempted through the module system
    # (using a callable-module mapping defined in the configuration)
    required_callables = ["echo"]
    """Provide custom install instructions for dependencies here"""

    # Python packages that will be required inside the `run` method.
    # This is just used for checking dependencies before running the node.
    required_packages = ["scipy"]

    # Make sure to specify types!
    out: Output[int] = Output()
    """
    Document all IO by placing a docstring under the attribute.
    Sphinx will detect this and generate pretty documentation :)

    """

    data: Parameter[int] = Parameter(default=42)
    """You can do the same for parameters! Default values will be added automatically."""

    def run(self) -> None:
        # Place any imports not contained in the standard maize environment in `run()`!
        import scipy
        self.logger.info(scipy.__version__)
        self.out.send(self.data.value)


# You should always write a test for your node:
@pytest.mark.skip(reason="Just an example")
class TestSuiteExample:
    def test_example(self) -> None:
        """Test our step in isolation"""
        rig = TestRig(Example)
        res = rig.setup_run(parameters={"data": 17})
        assert res["out"].get() == 17
