Custom workflows
================

Creating custom workflows and graphs encapsulating common behaviour can be done by adding modules to the ``maize/graphs`` namespace. Create an appropriate file and define your graphs as described in the :ref:`user guide <custom-graphs>`. For workflows usable from the commandline, define your workflow inside a function taking no parameters and no return (see the :ref:`user guide on workflows <custom-workflows>`) and call :func:`maize.utilities.io.setup_workflow` with your defined workflow. This will instantiate a commandline parser and expose all workflow parameters as arguments and flags. To make it callable anywhere, add this function to ``setup.cfg``.
