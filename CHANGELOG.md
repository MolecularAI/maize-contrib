# CHANGELOG

## Version 0.5.7

### Breaking changes
- Renamed some parameters for compatibility with new node tagging system

### Features
- Allow PDBQT for GNINA receptor
- Added tags to all nodes

### Changes
- Modified behaviour of `Mol2Mol`
- Modifications to allow use of `Gaussian` and `Crest` nodes independently from reaction control
- Sanitization for ADV is now active again by default
- Keep hydrogens when reading a molecule from an SDF block by default
- Increased default Schrodinger polling intervals
- Additional debug info for Schrodinger job submission, increased wait for jobserver restart
- Added logging of ports in use by Schrodinger job servers
- Schrodinger nodes now set additional env variable for local job server

### Fixes
- Don't use MPS with GNINA if we don't want to use the GPU
- Fixed Vina test failure due to missing properties
- Skip non-created input.pdbqt files for Vina
- Added explicit job server directory cleanup for Schrodinger

## Version 0.5.6

### Features
- Added AEV-PLIG ligand-protein complex scoring
- Added absolute solvation free energy node using OpenFE

### Changes
- GNINA node now adds hydrogens explicitly post-docking
- Added `query_interval` parameter to Schrodinger workflows

### Fixes
- Increased default `query_interval` value to avoid crashes on very long jobserver wait times

## Version 0.5.5

### Features
- Added PoseStability node using a simple MD simulation
- Added GNINA + pose stability workflows

### Changes
- Added option to specify GNINA reference by name

### Fixes
- Fixed incomplete scalar check when setting scores

## Version 0.5.4

### Features
- Added cofactor option to OpenFE

### Changes
- Refactored GNINA into separate single and ensemble nodes
- Made Ligprep more robust for bad smiles codes

### Fixes
- Fixed reversed merging direction for merge_libraries
- Only run GNINA with MPS locally, not on batch system
- Fixed unbound local for result checks
- Fixed GNINA not using GPUs correctly when running on multi-GPU nodes

## Version 0.5.3

### Features
- Added YAML workflows
- Added early termination option to OpenFE
- Added option to specify AL n_oracle as a percentage
- Added complex output to VinaFlex
- Added GPU flag to GNINA
- Added GPU / MPS check for GNINA
- Added tag arithmetic / aggregation nodes (TagAgg, TagMath)
- Improved node doc rendering
- Added SetName node
- Added isomer extraction node
- Added max_score parameter to Glide + associated workflows
- Added all GNINA scores
- Added GNINA ensemble docking

### Changes
- Suppress all NaN warnings for aggregated scores
- Downgraded Schrodinger download warning to debug
- Update for OpenFE 1.0
- Added explicit path inheritance to OpenFE
- ReinventExit node now returns names
- Removed GNINA pre-alignment
- Overhauled library merging
- Updated Glide REINVENT workflow to avoid writing out un-docked and extraneous poses

### Fixes
- Fixed REINVENT readlog printing everything when nothing changed
- Fixed bad transform access in OpenFE parsing
- Fixed missing mapping type in DynamicReference
- Made isomer name logging consistent (fixes #60)
- Fixed incomplete parsing of PDBQT output (#63)
- Improved constraint type input for glide, exposed on the workflow and also allow to be set from keywords
- Fixed incorrect filename incrementing behaviour in SaveSingleLibrary
- Fixed off-by-one in Ligprep for weird smiles
- Fix for Glide score parsing sometimes returning ints
- Fixed BestConformerFilter to sort NaN scores correctly

## Version 0.5.2

### Changes
- Improved node documentation
- SaveSingleLibrary now increments filenames by default

### Fixes
- Fixed Glide attempting to update with potentially non-existent tag
- Fixed Glide conformer sorting

## Version 0.5.1

### Features
- Added neighbor tags to OpenFE results
- Added pre-alignment to FEPGNINA subgraph
- Added ROCSish FlexibleShapeAlign node

### Changes
- ROCS no longer has a scores_only output
- Added Schrodinger job server query interval parameter
- Major documentation overhaul

### Fixes
- Added stopped status for Schrodinger job server
- Fixed error in array property conversion
- Tentative fix for Reinvent worker not shutting down properly
- Fix for BestIsomerFilter using incorrect descending value

## Version 0.5.0

### Features
- Overhauled scoring (now allows multiple scores to be set)
- Added tentative flexible PDBQT prep
- Added BestConformerFilter, removed RMSDFilter
- Added dynamic FEP references
- Added redundant-minimal mapping for OpenFE
- Added LigFilter to LigPrep
- Added pre-alignment option to GNINA
- Added LoadSingleRow function node
- Added scheduled epsilon greedy acquisition
- Improvements to ReactionControl

### Changes
- Exposed lambdas + replicas in FEP subgraph
- Improved Glide output parsing
- Removed InChI saving split strategy

### Fixes
- Fixed BestIsomerFilter not handling empty collections
- Fixed Glide failing on non-existent docking score
- Fix for merge_libraries not merging base keys
- LoadLibrary no longer renumbers atoms by default
- Fixed potentially inconsistent atom numbering in parsed conformers
- Documentation fixes
- Fixed EpsilonGreedy failing when n_random > n_best

## Version 0.4.4

### Features
- Added FEP Mapping subgraph
- Added Kartograf mapping backend for OpenFE
- Allowed multiple target charges in ChargeFilter
- Added SMILES tag to Ligprep output
- Added SaveSingleLibrary to save mols into single SDF

### Changes
- Allowed SetScoreTag to operate on the existing score tag
- Explicit cast to str when writing Isomer tags to SDF
- Simplified InChI-based library IO
- Exposed CNN scoring options for GNINA
- Glide now sends NaN on failure by default
- Updated Schrodinger host + n_jobs handling
- Redesigned additional Glide constraint inputs
- Schrodinger job submission overhaul, will now use a global jobserver in a temp folder
- Simplified BestIsomerFilter, with guarantee of only returning single Isomer

### Fixes
- Moved OFE mapping section to after protein receive to avoid deadlocks
- Removed Glide output atom renumbering to avoid incorrect isomer detection
- Fixed score aggregation over isomers failing if some have NaN scores

## Version 0.4.4

### Features
- Added `GNINA` node
- Added simple switch to AL
- Added generic tagging node
- Added mapping-only option to `OpenRFE`
- Added `ChargeFilter`
- Improvements to `Mol2MolStandalone`
- FEP subgraph improvements

### Changes
- Lots of extra logging
- Changed IC smiles attribute to property with tag
- Explicit casting to floats to avoid scorers sneaking in bad types
- Removed deprecated `super().prepare()` calls
- Removed references to old `MultiParameter`

### Fixes
- Fixed `EpsilonGreedy` not removing random component from surrogate
- Fix semi-rare PDBQT conversion failures
- Fixed OpenFE mapping failure recovery
- Fix for `EpsilonGreedy` failing when `n_mols` < `n_oracle`
- Fix for possible interference between RDKit and Qptuna

## Version 0.4.3

### Features
- Added FEP subgraph
- Added single-run AL progress node
- Added docs for standalone REINVENT use

### Fixes
- Fixed some minor type errors in AL
- Changed GlideDocking subgraph to use `File2Molecule`
- Include package-based config by default

## Version 0.4.2

### Features
- Added single-run active learning graph
- Added custom tag option to `ReinventExit`
- Added score_tag setter
- Added alternative aggregation to mol scoring
- Added timeout option to `RMSD`
- Added file-based score caching

### Changes
- Skip licadmin on execution failure
- Added retries to Schrodinger `TokenGuard`
- Added explicit cast of tags to strings in `save_sdf`

### Fixes
- Added catch for semi-rare meeko PDBQT parsing issue

## Version 0.4.1

### Changes
- Added File2Molecule to handle LoadMolecule with input

### Fixes
- Typing fixes
- Fixed TagSorter not using correct test runner
- Fixed first-conformer parsing problem in Glide

## Version 0.4.0
### Features
- Added Gromacs module (Lily)
- Added flexible docking with Vina
- Added additional tag handling utilities
- Added explicit prior and agent parameters to Reinvent
- Added `IsomerCollection` tagging system
- Added active learning module, subgraph, and example notebook

### Changes
- Reorganised cheminformatics

### Fixes
- Fixed relative-absolute conversion of BFEs not using reference correctly
- Fixed `MakeAbsolute` not handling disconnected subgraphs
- Fixed `SaveLibrary` not saving all conformers
- Fixed `AutoDockGPU` not detecting tarred grids as inputs
- Fixed OpenFE dumping failing on existing dump

## Version 0.3.2
### Features
- Tests can now be skipped automatically based on available config options
- Added absolute free energy conversion for `OpenRFE`

### Changes
- Updated REINVENT implementation
- Updated bundled maize

## Version 0.3.1
### Changes
- Updated bundled maize

### Fixes
- Fixed SaveCSV not using the correct tag order
- Various type fixes

## Version 0.3.0
### Features
- Several new filters
- More subgraphs
- OpenFE node
- Constraints for AutoDockGPU and Glide
- Nodes for reaction prediction
- Batched CSV saving
- Flexible docking with Vina

### Changes
- Various improvements to RMSD filtering
- Performance improvements for `Isomer` tag handling
- Allowed certain RDKit ops to timeout
- More robust Schrodinger job handling
- Integration tests
- Maize core wheel for automated builds

### Fixes
- Fix for zombie threads when using REINVENT
- Fix for installation only being possible using `-e`
- Cleaned up typing

## Version 0.2.3
### Changes
- Removed interface for REINVENT 3.2

### Fixes
- Cleanup of package structure
- Updated dependencies
- Added explicit check for Schrodinger license

## Version 0.2.2
### Features
- Added Schrodinger grid preparation tools

### Changes
- Adjusted timeout for Gypsum-DL
- Various enhancements for GLIDE

### Fixes
- Fixed `Vina` command validation for newer `Vina` versions
- Fixed parsing issues for `Vina`


## Version 0.2.1
### Features
- Improved REINVENT interface, using `ReinventEntry` and `ReinventExit`

### Changes
- Timeouts for conformer / isomer generation
- Improved logging

### Fixes
- Check for zero bytes for `Vina` output

## Version 0.2.0

Updated for public release.

## Version 0.1

Initial release.