.. maize-contrib documentation master file, created by
   sphinx-quickstart on Wed Mar  8 17:24:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

maize-contrib
=============
*maize* is a graph-based workflow manager for computational chemistry pipelines. This repository contains namespace packages allowing domain-specific extensions and steps for *maize*. You can find the core maize documentation `here <https://molecularai.github.io/maize>`_.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   docking

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Guide

   custom-nodes
   custom-workflows

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Reference

   steps/index
   utilities

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Core

   Steps <https://molecularai.github.io/maize/docs/steps>
   Maize <https://molecularai.github.io/maize>

Installation
------------
You can install *maize-contrib* with:

.. code-block:: bash

   conda env create -f env-users.yml
   conda activate maize
   pip install -e ./

The first step will install maize and all required dependencies. If you plan on developing, you should use ``env-dev.yml`` instead. If you already have *maize* installed, you can clone add the additional dependencies and install as above:

.. code-block:: bash

   conda activate maize-dev
   conda install -c conda-forge rdkit scipy openbabel
   pip install -e ./

Configuration
-------------
Each step documentation will contain information on how to setup and run the node, as well as install the required dependencies. Dependencies can be managed in several ways, depending on the node and workflow you are running:

* Through a ``module`` system:

  Specify a module providing an executable in the ``config.toml`` (see :ref:`config-workflow`) file. This module will then be loaded in the process running the node.

* With a separate python environment:

  Some nodes will require custom python environments that are likely to be incompatible with the other environments. In those cases, the node process can be spawned in a custom environment. Note that this environment must still contain *maize*. Required custom environments can be found in the appropriate node directory.

* By specifying the executable location and possibly script interpreter. This can also be accomplished using ``config.toml`` (see :ref:`config-workflow` and the maize example).

Tips
----
* If you're unsure where to start with custom node, you can copy the example at ``maize/steps/mai/example`` as a template.
* For custom environments it will be easiest to clone the ``maize-dev`` environment:
  
  .. code-block:: shell
     
     conda create --name maize-new --clone maize-dev

Roadmap
-------
* Glide & Ligprep
* JSON interface for REINVENT

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
