Custom nodes
============
*maize-contrib* is set up as a namespace package, with all custom nodes under ``maize/steps`` in appropriate subfolders.

Structure
---------
To add your own nodes, find or create an appropriate subfolder under ``maize/steps/mai`` (or ``maize/steps/ext`` for projects external to the Molecular AI group). Add a python file with your custom node definition, and expose it using an ``__init__.py`` file. Every node should have its own test function as part of the same module, and all test data should be contained in the ``data`` child directory.

Node creation
-------------
Creating custom nodes is described in the :ref:`user guide <custom-nodes>`. Nodes made available through *maize-contrib* should be documented using the *reStructuredText* (reST) syntax (`see the Sphinx docs <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ for more information) and contain references to the literature if appropriate. A common strategy will be wrapping existing scripts that may require custom modules not part of the maize core. In this case, creating an additional file with this code and importing it in the :meth:`~maize.core.node.Node.run` method of your node is a useful pattern.

If you require ``rdkit`` functionality, you should probably use the :mod:`~maize.utilities.chem` module instead. The included :class:`~maize.utilities.chem.chem.Isomer` class is a thin wrapper around the ``rdkit.Chem.Mol`` object, providing improved error handling and a more pythonic interface. Feel free to add additional functionality to the module as required.

Testing
-------
All tests are implemented using `pytest <https://docs.pytest.org/en/7.2.x/index.html>`_. Every node should have a test function in the same file. If there are multiple related tests, they should ideally be combined in a single test class. Some useful predefined fixtures are ``temp_working_dir``, to perform the test in a temporary directory and ``test_config`` for a configuration file (specified as an additional pytest option in ``pyproject.toml``).

To create a test for a node, use the :class:`~maize.utilities.testing.TestRig` utility class to wrap your node (with a custom configuration). You can then use the :meth:`~maize.utilities.testing.TestRig.setup_run` convenience method to specify both parameters and inputs to your node. Note that ``inputs`` takes a list of items that will be sent to the node separately, to allow for tests of nodes that read from the same input multiple times. Node results can then be retrieved by accessing the corresponding output in the returned dictionary and calling the :meth:`~maize.utilities.testing.MockChannel.get` method:

.. code-block:: python

   def test_node(temp_working_dir, test_config):
       rig = TestRig(YourNode, config=test_config)
       result = rig.setup_run(parameters={"value": 42}, inputs={"inp": [17]})
       assert result["out"].get() == 59
