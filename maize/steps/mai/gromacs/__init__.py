"""
This Python package offers functionalities for preparing the filesystem necessary
for a GROMACS Molecular Dynamics (MD) run. Additionally, it includes features that
fulfill the requirements of maize.

"""
from .file_utils import SaveFilesFromDict, read_sdf_and_save_mols
from .gmx import MDs


__all__ = ["SaveFilesFromDict", "MDs", "read_sdf_and_save_mols"]
