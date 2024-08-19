"""
comp chem Utils

Collection of Classes and Functions to support the processing, running and analysis 
of different types of computational chemistry calculations.

"""
# Code written by Mikhail Kabeshov, 2022.


import os
import math
import io
import logging
import numpy as np
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Union, TextIO
from typing_extensions import TypedDict
from pathlib import Path
from maize.utilities.chem.chem import Conformer



AtomEntry = TypedDict("AtomEntry", {"element": str, "atom_id": int, "coords": list[float]})
ConfTag = TypedDict("ConfTag", {"atoms": list[AtomEntry], "energy": float, "gradient": float})





log = logging.getLogger("run")

atom_string: list[str] = [
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",  # list of atoms corresponding
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Ni",
    "Ir",
]

atom_list: dict[int, str] = {
    1: "H",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",  # list of atoms corresponding
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    3: "Li",
    35: "Br",
    53: "I",
    46: "Pd",
    29: "Cu",
    28: "Ni",
    77: "Ir",
}

Atom_radii: dict[str, float] = {
    "N": 1.89,
    "H": 1.2,
    "He": 0.31,
    "Li": 1.82,
    "Be": 1.12,
    "Ne": 0.38,
    "Na": 1.9,
    "Mg": 1.45,
    "Al": 1.18,
    "Si": 1.11,
    "Ar": 0.71,
    "K": 2.43,
    "Ca": 1.94,
    "Sc": 1.84,
    "Ti": 1.76,
    "V": 1.71,
    "Cr": 1.66,
    "Mn": 1.61,
    "Co": 1.52,
    "Ni": 1.49,
    "Cu": 1.45,
    "Zn": 1.42,
    "Ga": 1.36,
    "Ge": 1.25,
    "As": 1.14,
    "Se": 1.03,
    "Kr": 0.88,
    "Rb": 2.65,
    "Sr": 2.19,
    "Y": 2.12,
    "Zr": 2.06,
    "Nb": 1.98,
    "Mo": 1.9,
    "Tc": 1.85,
    "Rh": 1.73,
    "Pd": 1.69,
    "Ag": 1.65,
    "Cd": 1.61,
    "In": 1.56,
    "Sn": 1.45,
    "Sb": 1.33,
    "Te": 1.23,
    "Xe": 1.08,
    "Cs": 2.98,
    "Ba": 2.53,
    "La": 2.51,
    "Ce": 2.49,
    "C": 1.85,
    "O": 2.294,
    "I": 1.15,
    "Cl": 2.38,
    "Ru": 1.78,
    "Br": 0.94,
    "F": 1.73,
    "P": 0.98,
    "S": 0.88,
    "B": 0.87,
    "Fe": 1.56,
    "Ir": 2.34,
    "Pt": 1.86,
    "Lu": 1.95,
    "Yb": 1.93,
    "Er": 1.94,
    "Sm": 1.91,
    "Os": 2.00,
    "W": 1.95,
    "Au": 2.05,
}  # double-check Ir, Lu, Yb, Er, Sm and Pt


Atom_mass: dict[str, float] = {
    "H": 1.008,
    "He": 4.003,
    "Li": 6.94,
    "Be": 9.012,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 98,
    "Ru": 101.07,
    "Rh": 102.906,
    "Pd": 106.42,
    "Ag": 107.868,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.710,
    "Sb": 121.760,
    "Te": 127.60,
    "I": 126.904,
    "Xe": 131.293,
    "Cs": 132.905,
    "Ba": 137.327,
    "La": 138.905,
    "Ce": 140.116,
    "Pr": 140.908,
    "Nd": 144.242,
    "Pm": 145,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.925,
    "Dy": 162.500,
    "Ho": 164.930,
    "Er": 167.259,
    "Tm": 168.934,
    "Yb": 173.045,
    "Lu": 174.967,
    "Hf": 178.486,
    "Ta": 180.948,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.967,
    "Hg": 200.592,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.980,
    "Po": 209,
    "At": 210,
    "Rn": 222,
    "Fr": 223,
    "Ra": 226,
    "Ac": 227,
    "Th": 232.038,
    "Pa": 231.036,
    "U": 238.029,
    "Np": 237,
    "Pu": 244,
    "Am": 243,
    "Cm": 247,
    "Bk": 247,
    "Cf": 251,
    "Es": 252,
    "Fm": 257,
    "Md": 258,
    "No": 259,
    "Lr": 262,
    "Rf": 267,
    "Db": 268,
    "Sg": 269,
    "Bh": 270,
    "Hs": 269,
    "Mt": 278,
    "Ds": 281,
    "Rg": 282,
    "Cn": 285,
    "Nh": 286,
    "Fl": 289,
    "Mc": 289,
    "Lv": 293,
    "Ts": 294,
    "Og": 294,
}  # double-check Ir, Lu, Yb, Er, Sm and Pt


""" 
Data taken from: 
International tables for Crystallography (2006), Vol. C, Chapter 9.5, pp. 790 - 811 
"""
Connectivity: dict[str, list[float]] = {
    "CC": [1.65, 1.439, 1.342, 1.20],
    "BB": [1.77],
    "BBr": [2.1],
    "BC": [1.68],
    "BCl": [1.85],
    "BI": [2.25],
    "BF": [1.38],
    "BN": [1.61],
    "BO": [1.48],
    "BP": [1.93],
    "BS": [1.93],
    "BrBr": [2.54],
    "BrC": [1.97],
    "BrI": [2.70],
    "BrN": [1.85],
    "BrO": [1.59],
    "BrP": [2.37],
    "BrS": [2.45],
    "BrSe": [2.62],
    "BrSi": [2.28],
    "BrTe": [3.1],
    "CCl": [1.74],
    "CF": [1.43],
    "CH": [1.15],
    "CI": [2.192],
    "CN": [1.56, 1.361, 1.315, 1.15],
    "CO": [1.50, 1.31, 1.28, np.nan],
    "CP": [1.93],
    "CS": [1.86, 1.724, 1.69, np.nan],
    "CSe": [1.97],
    "CSi": [1.89],
    "CTe": [2.16],
    "ClCl": [2.31],
    "ClI": [2.58],
    "ClN": [1.76],
    "ClO": [1.42],
    "ClP": [2.05],
    "ClS": [2.1],
    "ClSe": [2.25],
    "ClSi": [2.1],
    "ClTe": [2.52],
    "FN": [1.41],
    "FP": [1.715],
    "FS": [1.64],
    "FSi": [1.70],
    "FTe": [2.0],
    "HN": [1.1],
    "HO": [1.1],
    "II": [2.93],
    "HS": [1.45],
    "IN": [2.25],
    "IO": [2.17],
    "IP": [2.50],
    "IS": [2.75],
    "ITe": [2.95],
    "NN": [1.45, 1.32, 1.24, 1.10],
    "NO": [1.46, np.nan, 1.24, np.nan],
    "NP": [1.73, np.nan, 1.62, np.nan],
    "NS": [1.71, np.nan, 1.56, np.nan],
    "NSe": [1.86, np.nan, 1.795, np.nan],
    "NSi": [1.78],
    "NTe": [2.25],
    "OO": [1.50],
    "OP": [1.70, np.nan, 1.51, np.nan],
    "OS": [1.60, np.nan, 1.44, np.nan],
    "OSe": [2.0, np.nan, 1.60, np.nan],
    "OSi": [1.70],
    "OTe": [2.14],
    "PP": [2.26, np.nan, 2.05, np.nan],
    "PS": [2.30, np.nan, 1.95, np.nan],
    "PSe": [2.45, np.nan, 2.10, np.nan],
    "PSi": [2.26],
    "PTe": [2.50, np.nan, 2.35, np.nan],
    "SS": [2.10],
    "SSe": [2.20],
    "SSi": [2.15],
    "STe": [2.45],
    "SeSe": [2.35],
    "SeTe": [2.58],
    "SiSi": [2.40],
    "TeTe": [2.75],
    "FeH": [1.61],
    "ZnH": [1.62],
    "MoH": [1.69],
    "RhH": [1.58],
    "TaH": [1.78],
    "WH": [1.74],
    "ReH": [1.69],
    "OsH": [1.66],
    "IrH": [1.61],
    "PtH": [1.61],
    "ScB": [2.53],
    "TiB": [2.385],
    "CrB": [2.36],
    "CoB": [2.245],
    "NiB": [2.19],
    "CuB": [2.40],
    "ZnB": [2.27],
    "YB": [2.68],
    "ZrB": [2.34],
    "MoB": [2.47],
    "RuB": [2.47],
    "OsB": [2.31],
    "UB": [2.65],
    "VB": [2.36],
    "MnB": [2.31],
    "FeB": [2.28],
    "RhB": [2.39],
    "PdB": [2.26],
    "AgB": [2.53],
    "WB": [2.42],
    "ReB": [2.35],
    "IrB": [2.45],
    "PtB": [2.29],
    "AuB": [2.26],
    "HgB": [2.52],
    "HgC": [2.58],
    "RuC": [2.325],
    "OsC": [2.28],
    "RhC": [2.26],
    "CrC": [2.30],
    "MoC": [2.42],
    "TaC": [2.41],
    "WC": [2.42],
    "ReC": [2.17],
    "FeC": [2.17],
    "CoC": [2.16],
    "VC": [2.36],
    "PdC": [2.38],
    "IrC": [2.42],
    "PtC": [2.21],
    "MnC": [2.16],
    "NiC": [2.12],
    "CuC": [2.32],
    "ZnC": [2.25],
    "AgC": [2.43],
    "AuC": [2.22],
    "NbC": [2.33],
    "PrC": [2.80],
    "TiC": [2.48],
    "ZrC": [2.52],
    "TcC": [1.92],
    "HfC": [2.52],
    "LuC": [2.78],
    "ThC": [2.82],
    "UC": [2.81],
    "YC": [2.77],
    "GdC": [2.75],
    "ErC": [2.74],
    "YbC": [2.72],
    "SmC": [2.88],
    "EuC": [2.83],
    "CeC": [2.79],
    "HgN": [2.67],
    "RuN": [2.31],
    "OsN": [2.24],
    "RhN": [2.36],
    "CrN": [2.32],
    "MoN": [2.45],
    "TaN": [2.42],
    "WN": [2.49],
    "ReN": [2.34],
    "FeN": [2.36],
    "CoN": [2.24],
    "VN": [2.30],
    "PdN": [2.17],
    "IrN": [2.18],
    "PtN": [2.25],
    "MnN": [2.40],
    "NiN": [2.22],
    "CuN": [2.32],
    "ZnN": [2.32],
    "AgN": [2.46],
    "AuN": [2.17],
    "NbN": [2.42],
    "PrN": [2.73],
    "TiN": [2.32],
    "ZrN": [2.45],
    "TcN": [2.50],
    "HfN": [2.20],
    "LuN": [2.47],
    "ThN": [2.66],
    "UN": [2.66],
    "YN": [2.50],
    "GdN": [2.68],
    "ErN": [2.57],
    "YbN": [2.63],
    "SmN": [2.77],
    "EuN": [2.80],
    "CeN": [2.64],
    "CdN": [2.51],
    "DyN": [2.64],
    "NdN": [2.71],
    "HgO": [2.94],
    "RuO": [2.45],
    "OsO": [2.24],
    "RhO": [2.34],
    "CrO": [2.37],
    "MoO": [2.60],
    "TaO": [2.46],
    "WO": [2.25],
    "ReO": [2.40],
    "FeO": [2.36],
    "CoO": [2.36],
    "VO": [2.32],
    "PdO": [2.30],
    "IrO": [2.27],
    "PtO": [2.20],
    "MnO": [2.38],
    "NiO": [2.24],
    "CuO": [2.55],
    "ZnO": [2.45],
    "AgO": [2.70],
    "AuO": [2.20],
    "NbO": [2.37],
    "PrO": [2.58],
    "TiO": [2.25],
    "ZrO": [2.45],
    "TcO": [2.30],
    "HfO": [2.30],
    "LuO": [2.51],
    "ThO": [2.60],
    "UO": [2.65],
    "YO": [2.50],
    "GdO": [2.52],
    "ErO": [2.50],
    "YbO": [2.45],
    "SmO": [2.65],
    "EuO": [2.60],
    "CeO": [2.65],
    "CdO": [2.60],
    "DyO": [2.51],
    "NdO": [2.72],
    "TiF": [1.95],
    "VF": [1.95],
    "CrF": [1.95],
    "MnF": [1.95],
    "NiF": [2.11],
    "CuF": [2.00],
    "ZrF": [2.10],
    "NbF": [2.00],
    "MoF": [2.21],
    "WF": [2.10],
    "ReF": [2.05],
    "PtF": [2.05],
    "UF": [2.36],
    "AgF": [2.55],
    "MnSi": [2.57],
    "FeSi": [2.42],
    "MoSi": [2.61],
    "RuSi": [2.47],
    "RhSi": [2.38],
    "WSi": [2.59],
    "ReSi": [2.55],
    "OsSi": [2.43],
    "IrSi": [2.42],
    "PtSi": [2.39],
    "HgSi": [2.55],
    "HgP": [2.50],
    "RuP": [2.48],
    "OsP": [2.42],
    "RhP": [2.46],
    "CrP": [2.41],
    "MoP": [2.56],
    "WP": [2.58],
    "ReP": [2.50],
    "FeP": [2.40],
    "CoP": [2.47],
    "VP": [2.40],
    "PdP": [2.50],
    "IrP": [2.45],
    "PtP": [2.45],
    "MnP": [2.66],
    "NiP": [2.50],
    "CuP": [2.40],
    "AgP": [2.50],
    "AuP": [2.45],
    "NbP": [2.71],
    "TiP": [2.65],
    "ZrP": [2.85],
    "TcP": [2.53],
    "HfP": [2.70],
    "ThP": [3.15],
    "UP": [3.01],
    "CdP": [2.56],
    "HgS": [2.55],
    "RuS": [2.45],
    "OsS": [2.51],
    "RhS": [2.48],
    "CrS": [2.45],
    "MoS": [2.57],
    "TaS": [2.50],
    "WS": [2.55],
    "ReS": [2.50],
    "FeS": [2.61],
    "CoS": [2.51],
    "VS": [2.50],
    "PdS": [2.40],
    "IrS": [2.47],
    "PtS": [2.48],
    "MnS": [2.50],
    "NiS": [2.57],
    "CuS": [2.48],
    "ZnS": [2.47],
    "AgS": [2.70],
    "AuS": [2.40],
    "NbS": [2.74],
    "TiS": [2.55],
    "ZrS": [2.75],
    "TcS": [2.50],
    "LuS": [2.70],
    "ThS": [2.95],
    "US": [2.90],
    "CeS": [3.01],
    "CdS": [2.77],
    "DyS": [2.75],
    "HgCl": [2.70],
    "RuCl": [2.50],
    "OsCl": [2.45],
    "RhCl": [2.60],
    "CrCl": [2.40],
    "MoCl": [2.52],
    "TaCl": [2.63],
    "WCl": [2.60],
    "ReCl": [2.52],
    "FeCl": [2.44],
    "CoCl": [2.48],
    "VCl": [2.49],
    "PdCl": [2.52],
    "IrCl": [2.46],
    "PtCl": [2.51],
    "MnCl": [2.59],
    "NiCl": [2.47],
    "CuCl": [2.58],
    "ZnCl": [2.38],
    "AgCl": [2.72],
    "AuCl": [2.30],
    "NbCl": [2.57],
    "PrCl": [2.90],
    "TiCl": [2.55],
    "ZrCl": [2.64],
    "TcCl": [2.36],
    "HfCl": [2.44],
    "ThCl": [2.76],
    "UCl": [2.91],
    "YCl": [2.76],
    "ErCl": [2.62],
    "YbCl": [2.65],
    "CeCl": [2.64],
    "CdCl": [2.78],
    "RuAs": [2.48],
    "OsAs": [2.49],
    "RhAs": [2.50],
    "CrAs": [2.50],
    "MoAs": [2.62],
    "WAs": [2.56],
    "ReAs": [2.59],
    "FeAs": [2.38],
    "CoAs": [2.37],
    "VAs": [2.54],
    "PdAs": [2.48],
    "PtAs": [2.41],
    "MnAs": [2.53],
    "NiAs": [2.42],
    "CuAs": [2.75],
    "NbAs": [2.76],
    "TiAs": [2.70],
    "TcAs": [2.53],
    "CrSe": [2.56],
    "MnSe": [2.48],
    "FeSe": [2.43],
    "NiSe": [2.40],
    "CuSe": [3.11],
    "ZrSe": [2.68],
    "MoSe": [2.51],
    "RhSe": [2.47],
    "AgSe": [2.72],
    "WSe": [2.74],
    "ReSe": [2.60],
    "OsSe": [2.57],
    "IrSe": [2.55],
    "PtSe": [2.60],
    "HgSe": [2.65],
    "HgBr": [2.81],
    "RuBr": [2.60],
    "OsBr": [2.63],
    "RhBr": [2.63],
    "CrBr": [2.61],
    "MoBr": [2.73],
    "TaBr": [2.62],
    "WBr": [2.65],
    "ReBr": [2.68],
    "FeBr": [2.50],
    "CoBr": [2.41],
    "PdBr": [2.54],
    "IrBr": [2.60],
    "PtBr": [2.51],
    "MnBr": [2.70],
    "NiBr": [2.52],
    "CuBr": [2.60],
    "ZnBr": [2.45],
    "AgBr": [2.45],
    "AuBr": [2.45],
    "TiBr": [2.70],
    "TcBr": [2.50],
    "UBr": [2.85],
    "CdBr": [3.00],
    "CrTe": [2.81],
    "MnTe": [2.52],
    "FeTe": [2.58],
    "MoTe": [2.80],
    "PtTe": [2.58],
    "HgTe": [2.80],
    "HgI": [2.98],
    "RuI": [2.80],
    "OsI": [2.82],
    "RhI": [2.78],
    "CrI": [2.80],
    "MoI": [2.88],
    "WI": [2.90],
    "ReI": [2.85],
    "FeI": [2.70],
    "CoI": [2.75],
    "VI": [2.68],
    "PdI": [2.75],
    "IrI": [2.78],
    "PtI": [2.70],
    "MnI": [2.66],
    "NiI": [2.78],
    "CuI": [2.75],
    "ZnI": [2.65],
    "AgI": [2.92],
    "AuI": [2.86],
    "ZrI": [2.90],
    "YbI": [3.04],
    "CdI": [2.81],
    ##########
    "FeFe": [2.60],
    "IrIr": [2.90],
    "SiH": [1.60],
    "HH": [0.80],
    "BH": [],
    "AlAl": [3.05],
    "CrCr": [2.60],
    "IrRh": [3.00],
    "PtPt": [2.90],
    "PtRu": [2.85],
    "PH": [1.50],
}

Coordinative_bonds = {
    "CI": 2.50,
    "NH": 2.20,
    "OH": 2.20,
    "SH": 2.20,
    "CrH": 1.73,
    "FeH": 1.68,
    "MoH": 1.85,
    "RuH": 1.79,
    "RhH": 1.87,
    "WH": 1.905,
    "ReH": 1.84,
    "OsH": 1.84,
    "IrH": 1.85,
    "PtH": 2.05,
    "CoH": 1.75,
    "NiH": 1.70,
    "CuH": 1.70,
    "ThH": 2.12,
    "FeC": 2.35,
    "RuC": 2.55,
    "PtC": 2.50,
    "ZnC": 2.50,
    "HgS": 3.15,
    "TaS": 2.85,
    "PdS": 2.80,
    "MoS": 2.70,
    "ZnS": 2.80,
    "HgCl": 3.10,
    "NbCl": 3.00,
    "TcCl": 2.70,
    "NiSe": 2.70,
    "HgSe": 3.00,
}
## check with mikhail if required
Computational_time = {"LEY-SL2": 36, "LEY": 12, "LAPKIN-SL3": 12, "LAPKIN-SL2": 36}

electronegativity = {"F": 4.0, "O": 3.5, "N": 3.0, "Cl": 3.2, "S": 2.6, "Br": 3.0, "I": 2.7}


heavy_atoms = ["I", "Br", "Cl", "Cu", "Pd", "Ni", "Ir"]  # list of heavy atoms

## check with mikhail if required
oniom_symbols = ["L", "H"]


gaussian_pseudo = {
    "I": "MWB46",
    "Cu": "MDF10",
    "Pd": "MWB28",
    "Br": "MWB28",
    "Rh": "MWB28",
    "K": "MWB10",
    "Ni": "MDF10",
    "Ir": "MWB60",
}  # Dictionary of heavy atoms for which pseudo-potentials
# are required, with their respective functional as value.


class Atom:

    """
    Class for storing information, handling and manipulating atoms.

    Parameters
    ----------
    label
        string that represents the element
    position
        list of float with coordinates of the atom
    """

    def __init__(self, label: str, position: list[float]) -> None:
        self.label: str = label
        self.position: list[float] = [float(position[0]), float(position[1]), float(position[2])]
        self.coord: EntryCoord = EntryCoord(element=self.label, coords=self.position)
        self.number = 0
        self.type: str | None = None
        self.identity: str | None = None
        self.aromaticity = False
        self.Hattached: list[Atom] = []
        self.connectivity_number: int | str | None = None
        self.radius = Atom_radii[label]
        self.oniom: list[str] = []

    def distance(self, atom2: "Atom") -> float:
        """
        Calculates distance between self and another atom object.
        Returns the distance as a float

        Parameters
        ----------
        atom2
            Atom to check the distance from self

        Returns
        -------
        float
            distance between the atoms0
        """

        return math.sqrt(
            (atom2.position[0] - self.position[0]) ** 2
            + (atom2.position[1] - self.position[1]) ** 2
            + (atom2.position[2] - self.position[2]) ** 2
        )

    def check_connected(self, atom2: "Atom") -> bool:
        """
        Checks if self is connected to another atom
        by evaluating the interatomic distance

        Parameters
        ----------
        atom2
            Atom object assumed to be connected with self

        Returns
        -------
        bool
            Boolean True if atom2 connected
        """
        distance_check = self.distance(atom2) < (self.radius + atom2.radius) * 0.55
        return distance_check


@dataclass
class EntryCoord:
    """
    Dataclass to handle atomic coordinates more clearly.

    Attributes
    ----------

    element
        string with the element label of the atom
    coords
        coordinates of the atom
    """

    element: str

    coords: list[float]

    def to_Atom(self) -> Atom:
        """
        Converts EntryCoord istance to an atom object

        Returns
        -------
        Atom
            Atom object istance
        """

        atom_obj = Atom(label=self.element, position=self.coords)

        return atom_obj


class Bond:
    """
    Class to store information about Bonds

    Parameters
    ----------
    atom1
        First atom participating to the bond
    atom2
        Second atom participating to the bond
    order
        Bond order
    """

    def __init__(self, atom1: Atom, atom2: Atom, order: int) -> None:
        self.order: int = order
        self.atom1: Atom = atom1
        self.atom2: Atom = atom2
        self.length: float = atom1.distance(atom2)


class Structure:
    """
    Class used to handle and manipulate molecules. Functions in this class
    can perform geometric operations, store data regarding the molecule
    as attributes and manipulate molecular structure.

    Parameters
    ----------
    atoms
        list of atom objects forming the molecule
    """

    def __init__(self, atoms: list[Atom]) -> None:
        self.atoms: list[Atom] = atoms
        self.charge: int | None = None
        self.multiplicity: int | None = None
        self.name: str | None = None
        self.energy: float | str | None = None
        self.connectivity: list[Bond] = []
        self.solvent = "thf"
        self.Ecorr = None
        self.retry = None
        self.bonds_to_keep: list[Bond] = []
        self.translation_vector: Union[str, list[str], None] = None
        self.mol_dict: list[EntryCoord] = [
            EntryCoord(element=i.label, coords=i.position) for i in self.atoms
        ]

    def add_atoms(self, atoms: list[Atom]) -> "Structure":
        """
        Returns a Structure object from a generic list of Atom objects

        Parameters
        ----------
        atoms
            list of Atom objects

        Returns
        -------
        Structure
            Structure istance with the new added atoms
        """

        self.atoms.sort(key=lambda x: x.number, reverse=False)
        atom_list: list[Atom] = self.atoms
        atom_number = len(atom_list) + 1
        for i in atoms:
            i.number = atom_number
            atom_list.append(i)
            atom_number += 1
        new_structure = Structure(atom_list)
        return new_structure

    def remove_atoms(self, atoms_to_remove: list[Atom]) -> "Structure":
        """
        Removes atoms from a Structure object.
        Returns the Structure object without the atoms to be removed

        Parameters
        ----------
        atoms_to_remove
            list of Atom objects to be removed

        Returns
        -------
        Structure
            Structure istance with selected atoms removed

        """
        self.atoms.sort(key=lambda x: x.number, reverse=False)
        atom_list: list[Atom] = self.atoms
        atom_list_remove: list[Atom] = []
        for i in atoms_to_remove:
            for j in atom_list:
                if i == j:
                    atom_list_remove.append(j)

        atom_list_new: list[Atom] = []
        for i in atom_list:
            if i not in atom_list_remove:
                atom_list_new.append(i)
        atom_list_new.sort(key=lambda x: x.number, reverse=False)
        atom_number = 1
        for ii in atom_list_new:
            ii.number = atom_number
            atom_number += 1
        structure_new = Structure(atom_list_new)
        structure_new.name = self.name
        return structure_new

    def Hconnected(self) -> None:
        """
        Auxillary function which finds protons
        """
        for j in self.atoms:
            if j.type == "heavy":
                Hlst: list[Atom] = []
                for jj in self.atoms:
                    if jj.type == "light":
                        if jj.check_connected(j):
                            Hlst.append(jj)
                j.Hattached = Hlst

    def gaussian_heavy_atoms(self) -> list[str]:
        """
        Checks heavy atoms in a molecule and returns them.
        Heavy atom defined as those included in the gaussian_pseudo list.

        Returns
        -------
        list[str]
            list of Heavy atoms included in the Structure object

        """
        heavy_atoms: list[str] = []
        for i in self.atoms:
            if i.label in gaussian_pseudo:
                if i.label not in heavy_atoms:
                    heavy_atoms.append(i.label)
        return heavy_atoms

    def gaussian_light_atoms(self) -> list[str]:
        """
        Checks light atoms in a molecule and returns them.
        All those not included in gaussian_pseudo list

        Returns
        -------
        light_atoms
            returns a list of light atoms included in the Structure object

        """

        light_atoms: list[str] = []
        for i in self.atoms:
            if i.label not in gaussian_pseudo:
                if i.label not in light_atoms:
                    light_atoms.append(i.label)
        return light_atoms

    def gaussian_link1(
        self,
        functional: str,
        basis: str,
        solvation: str,
        options: str,
        input_open: TextIO,
        solvent: str,
        maxsteps: int | None,
        modredundant: bool,
        oniom: str,
        multiplicity: int,
    ) -> None:
        """
        Function called by the main input file generator
        to add linked jobs to main g16 job.

        Parameters
        ----------
        functional
            functional to be added in the linked job
        basis
            basis set to be added in the linked job
        solvation
            class of implicit model to be used in the linked job
        options
            extra options to be included in the linked job
        input_open
            textIO wrapper for the input file
        solvent
            solvent to be used in the linked job
        maxsteps
            steps for modredun calculation
        modredundant
            Defines whether or not a calculation with redundant coordinate is requested
        oniom
            Option for oniom calculation
        multiplicity
            multiplicity to be used in the calculation

        """

        if self.name is None:
            raise NameError("Name of the file not found. Code cannot search the file without it")

        full_path = str(os.path.abspath(self.name))
        percentage = "%"
        
        f = input_open

        f.write("--Link1--\n")
        f.write(f"{percentage}mem=16000MB\n")
        f.write(f"{percentage}nprocshared=12\n")

       

        if maxsteps:
            f.write("%kjob l103 " + str(maxsteps) + "\n")
        if not len(self.gaussian_heavy_atoms()) == 0:
            f.write(
                "# "
                + self.functional(str(functional + "/genecp"), oniom)
                + " "
                + self.xqc(multiplicity)
                + " guess=read geom=allcheck "
                + options
                + " scrf=("
                + solvation
                + ",solvent="
                + solvent
                + ",dovacuum)\n"
            )
            f.write("\n")
        else:
            f.write(
                "# "
                + self.functional(str(functional + "/" + basis), oniom)
                + " guess=read "
                + self.xqc(multiplicity)
                + " geom=allcheck "
                + options
                + " scrf=("
                + solvation
                + ",solvent="
                + solvent
                + ",dovacuum)\n"
            )
            f.write("\n")
        if modredundant:
            nbondsfreeze = int(input("Please enter the number of bonds to freeze for " + full_path))
            i = 0
            atmfreeze: list[int] = []
            while i < nbondsfreeze:
                atmfreeze.append(
                    int(
                        input(
                            "Please enter the number of the first atom in bond "
                            + str(i + 1)
                            + " for "
                            + full_path
                        )
                    )
                )
                atmfreeze.append(
                    int(
                        input(
                            "Please enter the number of the second atom in bond "
                            + str(i + 1)
                            + " for "
                            + full_path
                        )
                    )
                )
                i += 1
            j = 0
            while j < 2 * nbondsfreeze:
                f.write("\n")
                f.write(str(atmfreeze[j]) + " " + str(atmfreeze[j + 1]) + " F")
                j += 2
            f.write("\n")
        if not len(self.gaussian_heavy_atoms()) == 0:
            light_atoms = ""
            for label in self.gaussian_light_atoms():
                light_atoms += label + " "
            light_atoms += str("0")

            f.write(light_atoms + "\n")
            f.write(basis + "\n")
            f.write("****\n")

            for label in self.gaussian_heavy_atoms():
                f.write(label + " 0\n")
                f.write(gaussian_pseudo[label] + "\n")
                f.write("****" + "\n")
            f.write("\n")
            for ii in self.gaussian_heavy_atoms():
                f.write(ii + " 0\n")
                f.write(gaussian_pseudo[ii] + "\n")
            f.write("\n")

    def functional(self, high_layer: str, low_layer: str) -> str:
        """
        Define layers for oniom calculations and returns appropriate annotation for input file

        Parameters
        ----------
        high_layer
            atoms to add in the high layer
        low layer
            atoms to add in the low layer

        Returns
        -------
        str
            list of atoms in the high layer
        str
            string with the composed list of atoms in the high and low layer
        """

        if low_layer == "N/A":
            return high_layer
        else:
            out_str = str("ONIOM(" + high_layer + ":" + low_layer + ")")
            return out_str

    def xqc(self, multiplicity: int) -> str:
        """
        Adds the XQC (Extra Quadratic Convergence) keyword to the SCF option in
        the gaussian calculation if heavy atoms are present in the input molecule

        Check  https://gaussian.com/scf/ for reference on SCF keyword on gaussian

        Parameters
        ----------
        multiplicity
            multiplicity associated with the molecule

        Returns
        -------
        str
            empty string if no heavy atom present
        str
            string with the appropriate keyword for XQC
        """

        if multiplicity == 1 and len(self.gaussian_heavy_atoms()) == 0:
            no_heavy = ""
            return no_heavy

        xqc_str = "scf=xqc"

        return xqc_str

    def export_g16_into_maize(
        self,
        charge: int,
        multiplicity: int,
        solvent: str,
        mem: str,
        nproc: str,
        mode: str,
        oniom: str,
        link: bool,
    ) -> str | None:
        """
        Creates a gaussian input file and returns the full path of the file as string.
        Type of calculation can be selected with the argument mode.

        Modes tested for now:
            * 'sp': single-point calculation


        Parmaters
        ---------

        charge
            charge assigned to the structure to calculate
        multiplicity
            multiplicity assigned to the structure to calculate
        solvent
            implicit solvent selected for the calculation
        mem
            allocated memory for the calculation
        nproc
            allocated processors for the calculation
        mode
            define the type of calculation to run
        oniom
            run oniom calculation

        Returns
        -------
        str
            string with path of the gaussian input file
        """

        if self.name is None:
            raise NameError("Name of the file not found. Code cannot search the file without it")

        full_path = str(os.path.abspath(self.name))
        new_input = full_path + ".in"

        log.info(full_path)
        memory_requirements = str(mem) + "MB"
        num_proc = str(nproc)
        percentage = "%"

        with open(new_input, "w") as g16_inp:
            g16_inp.write(f"{percentage}mem={memory_requirements}\n")
            g16_inp.write(f"{percentage}nprocshared={num_proc}\n")

            route_section: str = ""

            if mode == "minimum":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt "
                        + self.xqc(multiplicity)
                        + " freq scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "sp":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# "
                        + self.xqc(multiplicity)
                        + " scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# "
                        + self.xqc(multiplicity)
                        + " "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "minimum_flat":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt=(MaxStep=10) "
                        + self.xqc(multiplicity)
                        + " freq scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt=(MaxStep=10) "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "minimum_vacuum":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )

            elif mode == "minimum_NWChem":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("b3lyp/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("b3lyp/6-31g(d,p)", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )

            elif mode == "minimum_polymer":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("hseh1pbe/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("hseh1pbe/6-31g(d,p)", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )

            elif mode == "minimum_modredundant":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# freq "
                        + self.xqc(multiplicity)
                        + " "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )
                else:
                    route_section = (
                        "# freq "
                        + self.xqc(multiplicity)
                        + " "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "TS":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt=(ts,calcfc,noeigentest) "
                        + self.xqc(multiplicity)
                        + " freq scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt=(ts,calcfc,noeigentest) "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "IRC":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# irc=(maxpoints=20,calcfc) "
                        + self.xqc(multiplicity)
                        + " scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# irc=(maxpoints=20,calcfc) "
                        + self.xqc(multiplicity)
                        + " "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "TS_cartesian":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt=(ts,cartesian,calcfc,noeigentest) "
                        + self.xqc(multiplicity)
                        + " freq scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt=(ts,cartesian,calcfc,noeigentest) "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "TS_initial":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# freq "
                        + self.xqc(multiplicity)
                        + " scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# freq "
                        + self.xqc(multiplicity)
                        + " "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "minimum_cartesian":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt=cartesian "
                        + self.xqc(multiplicity)
                        + " freq scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt=cartesian "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("wb97xd/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "TS_tuning":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt=(TS,MaxStep=10,calcall,noeigentest) "
                        + self.xqc(multiplicity)
                        + " freq scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt=(TS,MaxStep=10,calcall,noeigentest) "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            elif mode == "TS_flat":
                if not len(self.gaussian_heavy_atoms()) == 0:
                    route_section = (
                        "# opt=(TS,MaxStep=10,calcfc,noeigentest) "
                        + self.xqc(multiplicity)
                        + " freq scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum) "
                        + self.functional("MN15/genecp", oniom)
                        + " Integral(Grid=UltraFineGrid)\n"
                    )
                else:
                    route_section = (
                        "# opt=(TS,MaxStep=10,calcfc,noeigentest) "
                        + self.xqc(multiplicity)
                        + " freq "
                        + self.functional("MN15/cc-pvdz", oniom)
                        + " Integral(Grid=UltraFineGrid) scrf=(smd,solvent="
                        + solvent
                        + ",dovacuum)\n"
                    )

            g16_inp.write(route_section)
            g16_inp.write("\n")
            g16_inp.write("Title\n")
            g16_inp.write("\n")
            g16_inp.write(str(charge) + " " + str(multiplicity) + "\n")

            for j in self.atoms:
                oniom_layer = " "
                if not len(j.oniom) == 0:
                    for kk in j.oniom:
                        oniom_layer += kk + " "
                g16_inp.write(
                    j.label
                    + " "
                    + str("{0:.5f}".format(j.position[0]))
                    + " "
                    + str("{0:.5f}".format(j.position[1]))
                    + " "
                    + str("{0:.5f}".format(j.position[2]))
                    + oniom_layer
                    + "\n"
                )
            g16_inp.write("\n")

            if self.translation_vector is not None:
                g16_inp.write(
                    "TV "
                    + str(self.translation_vector[0])
                    + " "
                    + str(self.translation_vector[1])
                    + " "
                    + str(self.translation_vector[2])
                    + "\n"
                )
            if not len(self.gaussian_heavy_atoms()) == 0:
                light_atoms = ""
                for i in self.gaussian_light_atoms():
                    light_atoms += i + " "
                light_atoms += str("0")
                g16_inp.write(light_atoms + "\n")
                g16_inp.write("CC-PVDZ" + "\n")
                g16_inp.write("****" + "\n")

                for i in self.gaussian_heavy_atoms():
                    g16_inp.write(i + " 0\n")
                    g16_inp.write(gaussian_pseudo[i] + "\n")
                    g16_inp.write("****" + "\n")
                g16_inp.write("\n")
                for ii in self.gaussian_heavy_atoms():
                    g16_inp.write(ii + " 0\n")
                    g16_inp.write(gaussian_pseudo[ii] + "\n")
                g16_inp.write("\n")
            if mode == "TS_initial":
                self.gaussian_link1(
                    "MN15",
                    "cc-pvdz",
                    "smd",
                    (
                        "Integral(Grid=UltraFineGrid) OPT=(MaxCycles=500,ReadFC,ModRedundant, Tight)"
                        " IOP(1/8=15,1/9=1)"
                    ),
                    g16_inp,
                    solvent,
                    50,
                    True,
                    oniom,
                    multiplicity,
                )
                self.gaussian_link1(
                    "MN15",
                    "cc-pvdz",
                    "smd",
                    "OPT=(TS,NoEigenTest,MaxCycles=500,CalcFC,NoFreeze,Tight) IOP(1/8=10,1/9=1)",
                    g16_inp,
                    solvent,
                    15,
                    False,
                    oniom,
                    multiplicity,
                )
                self.gaussian_link1(
                    "MN15",
                    "cc-pvdz",
                    "smd",
                    "OPT=(TS,NoEigenTest,MaxCycles=500,CalcFC) Freq IOP(1/8=10,1/9=1)",
                    g16_inp,
                    solvent,
                    15,
                    False,
                    oniom,
                    multiplicity,
                )
                self.gaussian_link1(
                    "MN15",
                    "cc-pvdz",
                    "smd",
                    "OPT=(TS,NoEigenTest,MaxCycles=500,CalcFC) Freq IOP(1/8=10,1/9=1)",
                    g16_inp,
                    solvent,
                    None,
                    False,
                    oniom,
                    multiplicity,
                )
            elif mode == "minimum_modredundant":
                self.gaussian_link1(
                    "MN15",
                    "cc-pvdz",
                    "smd",
                    (
                        "Integral(Grid=UltraFineGrid) OPT=(MaxCycles=500,ReadFC,ModRedundant)"
                        " IOP(1/8=15,1/9=1)"
                    ),
                    g16_inp,
                    solvent,
                    None,
                    True,
                    oniom,
                    multiplicity,
                )

            if not mode == "minimum_polymer" and link:
                self.gaussian_link1(
                    "MN15",
                    "cc-pvtz",
                    "smd",
                    "Integral(Grid=UltraFineGrid)",
                    g16_inp,
                    solvent,
                    None,
                    False,
                    oniom,
                    multiplicity,
                )
                self.gaussian_link1(
                    "b3lyp",
                    "6-311+(d,p)",
                    "smd",
                    "Integral(Grid=UltraFineGrid)",
                    g16_inp,
                    solvent,
                    None,
                    False,
                    oniom,
                    multiplicity,
                )
                self.gaussian_link1(
                    "tpssh",
                    "cc-pvtz",
                    "smd",
                    "Integral(Grid=UltraFineGrid)",
                    g16_inp,
                    solvent,
                    None,
                    False,
                    oniom,
                    multiplicity,
                )
                self.gaussian_link1(
                    "m062x",
                    "cc-pvtz",
                    "smd",
                    "Integral(Grid=UltraFineGrid)",
                    g16_inp,
                    solvent,
                    None,
                    False,
                    oniom,
                    multiplicity,
                )
            g16_inp.write("\n")

        return new_input

    def connectivity_by_distance(self) -> None:
        """
        Populates connectivity of a Structure istance by checking interatomic distances

        """

        processed_atoms: list[Atom] = []
        bonds: list[Bond] = []
        for i in self.atoms:
            processed_atoms.append(i)
            for j in self.atoms:
                if j not in processed_atoms:
                    label: str = "N/A"
                    label1: str = i.label + j.label
                    label2: str = j.label + i.label
                    if label1 in Connectivity:
                        label = label1
                    if label2 in Connectivity:
                        label = label2
                    if not label == "N/A":
                        if len(Connectivity[label]) == 1:
                            if i.distance(j) < Connectivity[label][0]:
                                bond = Bond(i, j, 1)
                                bonds.append(bond)
                        elif len(Connectivity[label]) == 4:
                            if i.distance(j) < Connectivity[label][0]:
                                if not np.isnan(Connectivity[label][1]):
                                    if i.distance(j) > Connectivity[label][1]:
                                        bond = Bond(i, j, 1)
                                        bonds.append(bond)
                                    elif i.distance(j) < Connectivity[label][1]:
                                        if i.distance(j) > Connectivity[label][2]:
                                            bond = Bond(i, j, -1)
                                            i.aromaticity = True
                                            j.aromaticity = True
                                            bonds.append(bond)
                                        elif i.distance(j) < Connectivity[label][2]:
                                            if not np.isnan(Connectivity[label][3]):
                                                if i.distance(j) < Connectivity[label][3]:
                                                    bond = Bond(i, j, 3)
                                                    bonds.append(bond)
                                                elif i.distance(j) > Connectivity[label][3]:
                                                    bond = Bond(i, j, 2)
                                                    bonds.append(bond)
                                            else:
                                                bond = Bond(i, j, 2)
                                                bonds.append(bond)
                                else:
                                    if i.distance(j) < Connectivity[label][2]:
                                        if not np.isnan(Connectivity[label][3]):
                                            if i.distance(j) < Connectivity[label][3]:
                                                bond = Bond(i, j, 3)
                                                bonds.append(bond)
                                            elif i.distance(j) > Connectivity[label][3]:
                                                bond = Bond(i, j, 2)
                                                bonds.append(bond)
                                        else:
                                            bond = Bond(i, j, 2)
                                            bonds.append(bond)
                                    elif i.distance(j) > Connectivity[label][2]:
                                        bond = Bond(i, j, 1)
                                        bonds.append(bond)
                        if label in Coordinative_bonds:
                            if i.distance(j) < Coordinative_bonds[label]:
                                if i.distance(j) > Connectivity[label][0]:
                                    bond = Bond(i, j, -2)
                                    bonds.append(bond)
        self.connectivity = bonds

    def determine_atom_by_number(self, number: int) -> Atom:
        """
        Function that finds the atom object using its sequence number in the molecule object

        Parameters
        ----------
        number
            Number of the position in the sequence to search for

        Returns
        -------
        Atom
            Atom object matching the sequence number
        """

        result = Atom("C", [0.0, 0.0, 0.0])
        for i in self.atoms:
            if i.number == number:
                result = i
        return result

    def write_xyz(self, path: Path) -> None:
        """
        this functions writes a Structure object as a XYZ file.
        The name of the file will be the name assigned to the object.

        Parameters
        -----
        path
            Path of the xyz file to write
        """
        if self.name is None:
            name_short = "default name"
        else:    
            name_short = str((os.path.basename(self.name)))
            
        number = len(self.atoms)
        xyz_file = path

        with open(xyz_file, "w") as input_open:
            input_open.writelines(str(number) + "\n" + str(name_short) + "\n")
            for j in self.atoms:
                input_open.write(
                    j.label
                    + " "
                    + str("{0:.5f}".format(j.position[0]))
                    + " "
                    + str("{0:.5f}".format(j.position[1]))
                    + " "
                    + str("{0:.5f}".format(j.position[2]))
                    + "\n"
                )

    def get_mw(self) -> float:
        """
        Calculates the molecular weight of the structure by enumerating the atoms
        and adding the individual atomic masses.

        """
        mw = 0.0

        for atom in self.atoms:
            element = atom.label
            atm_mass = Atom_mass[element]
            mw += atm_mass

        return mw


class g_Output:

    """
    Class containing functions to process and analyze Gaussian16 output files.

    Parameters
    ----------
    filename
        Gaussian output file

    """

    def __init__(self, filename: str) -> None:
        self.filename = filename

    def final_energy(self) -> float | str | None:
        """
        Get the final single point energy of the gaussian calculation

        Returns
        -------
        float | str
            single point energy value
        """
        energy = None

        with open(self.filename, "r") as f:
            for line in f.readlines():
                if "SCF Done:" in line:
                    line_split = line.split()
                    pos = line_split.index("=") + 1
                    energy = float(line_split[pos])

        if energy is not None:
            return energy
        return None

    def gibbs_correction(self) -> float | str:
        """
        Returns Gibbs free energy correction if thermodynamic results are available.

        Returns
        -------
        float | str
            Correction to the gibbs free energy
        """

        enthalpy_correction = None
        entropy = None

        with open(self.filename, "r") as f:
            for line in f.readlines():
                if "correction to Enthalpy" in line:
                    line_split = line.split()
                    pos = line_split.index("=") + 1
                    enthalpy_correction = float(line_split[pos]) / 627.503
                elif "Total Entropy" in line:
                    line_split = line.split()
                    pos = line_split.index("=") + 1
                    entropy = float(line_split[pos]) / 627.503

        if enthalpy_correction is not None and entropy is not None:
            gibbs_correction = enthalpy_correction - 298 * entropy / 1000
            return gibbs_correction
        else:
            return "N/A"

    def solvent_output(self) -> str:
        """
        Returns the implicit solvent used in the calculation

        Returns
        -------
        str
            string with the solvent used
        """
        solvent = "N/A"
        with open(self.filename, "r") as f:
            for line in f.readlines():
                if "Solvent " in line:
                    line_split = line.split()
                    solvent = line_split[2]
        return solvent

    def normal_termination(self) -> bool:
        """
        Checks for normal termination of gaussian calculation.

        Returns
        -------
        bool
            Bool value. True if the calculation finished correctly.
        """

        check = False
        with open(self.filename, "r") as f:
            lines = f.readlines()
            if "Normal termination of Gaussian" in lines[-1]:
                check = True
        return check

    def frequency_analysis_gaussian(self) -> str:
        """
        Checks for the presence of negative frequencies.
        Assess if geometry is a minimum or saddle on the PES.

        Returns
        -------
        str
            string containing the nature of the point on the PES
        """

        status = "N/A"
        if self.normal_termination():
            with open(self.filename, "r") as f:
                for line in f.readlines():
                    if "imaginary frequencies (negative Signs)" in line:
                        status = "saddle"
                    else:
                        status = "minimum"
        return status

    def charge_gaussian(self) -> int | None:
        """
        Returns the charge used in the calculation

        Returns
        -------
        int
            string with charge value
        """

        charge = None
        with open(self.filename, "r") as f:
            for line in f.readlines():
                if "Charge = " in line:
                    line_split = line.split()
                    pos = line_split.index("Charge") + 2
                    charge = int(line_split[pos])
        return charge

    def multiplicity_gaussian(self) -> int | None:
        """
        Returns the multiplicity used in the calculation

        Returns
        -------
        int
            string with multiplicity value
        """

        multiplicity = None

        with open(self.filename, "r") as f:
            for line in f.readlines():
                if "Multiplicity = " in line:
                    line_split = line.split()
                    pos = line_split.index("Multiplicity") + 2
                    multiplicity = int(line_split[pos])
        return multiplicity

    def error_check_gaussian(self) -> str:
        """
        Checks for error messages in Gaussian Calculations
        and returns an explanatory message. For a detailed list of recurrent
        gaussian error message please refer to:
        https://docs.alliancecan.ca/wiki/Gaussian_error_messages

        Returns
        -------
        str
            string for optimisation ran out of steps
        str
            string for bad initial geometry
        str
            string for unindentified error
        """
        steps_msg = "steps_exceeded"
        geom_msg = "bad_geometry"
        generic = "unidentified_error"

        with open(self.filename) as f:
            if "Error termination via Lnk1e in /usr/local/Cluster-Apps/gaussian/g09/l9999.exe" in (
                list(f)[-3]
            ):
                return steps_msg

            elif "Error termination via Lnk1e in /usr/local/Cluster-Apps/gaussian/g09/l1.exe" in (
                list(f)[-3]
            ):
                return steps_msg

            elif "Error termination via Lnk1e in /usr/local/Cluster-Apps/gaussian/g09/l103.exe" in (
                list(f)[-3]
            ):
                return geom_msg
            elif "Error termination via Lnk1e in /usr/local/Cluster-Apps/gaussian/g09/l502.exe" in (
                list(f)[-3]
            ):
                return geom_msg
            else:
                return generic

    def intermediate_output(self) -> Structure:
        """
        Reads the output files and stores the resulting parameters
        from the calculation in a Structure object. Returns the Structure object
        populated with the calculations results.

        Stored parameters:
        * Molecular Geometry
        * Charge
        * Multiplicity
        * Solvent

        Returns
        -------
        Structure
            Strucutre object corresponding to the output geometry
        """

        atmlst: list[Atom] = []

        with open(self.filename, "r") as f:
            check1 = 0
            check2 = 0
            atmnumb = 1
            for line in f.readlines():
                if "Standard orientation:" in line:
                    check1 = 1
                    atmlst = []
                    check2 = 0
                    atmnumb = 1
                    continue
                elif check1 == 1:
                    if "---" in line:
                        check2 += 1
                        continue
                    elif check2 == 3:
                        check1 = 0
                        continue
                    elif check2 == 2:
                        line_split = line.split()
                        atom = Atom(
                            atom_list[int(line_split[1])],
                            [float(line_split[3]), float(line_split[4]), float(line_split[5])],
                        )
                        atmlst.append(atom)
                        if atom.label == "H":  # choosing atom type
                            atom.type = "light"
                        else:
                            atom.type = "heavy"
                        atom.number = atmnumb
                        atom.identity = atom.label + str(atom.number)
                        atmnumb += 1
                        continue
                else:
                    continue

        intermediate_out = Structure(atmlst)
        intermediate_out.name = str(self.filename)[:-4]
        intermediate_out.energy = self.final_energy()
        intermediate_out.solvent = self.solvent_output()
        intermediate_out.charge = self.charge_gaussian()
        intermediate_out.multiplicity = self.multiplicity_gaussian()
        intermediate_out.connectivity_by_distance()
        return intermediate_out


class Loader:

    """
    Class that groups various functions to load external files
    or data types with molecular information into Structure objects.

    Supported conversions:

    * sdf file to Structure
    * xyz file to Structure
    * Isomer to Structure
    * RDKIT MOL to Structure

    Parameters
    ----------
    source_filename
        input file with molecule data

    """

    def __init__(self, source_filename: str):
        self.source_filename = source_filename

    def molecule(self) -> Structure:
        """
        Helper function for creating a molecule object from an sdf file

        Returns
        -------
        Structure
            Structure object
        """
        atmlst: list[Atom] = []
        with open(self.source_filename, "r") as f:
            atmnumb = 1
            check1 = 1
            for line in f:
                if "END" in line:
                    check1 = 0
                elif "." in line and check1 == 1 and is_number(line.split()[0]):
                    lineproc = line.split()
                    lineproc[1] = label_format(lineproc[1])
                    atom = Atom(
                        lineproc[3], [float(lineproc[0]), float(lineproc[1]), float(lineproc[2])]
                    )

                    atom.connectivity_number = atom_string.index(atom.label)
                    atmlst.append(atom)
                    if atom.label == "H":  # choosing atom type
                        atom.type = "light"
                    else:
                        atom.type = "heavy"
                    atom.number = atmnumb
                    atmnumb += 1

        molecule = Structure(atmlst)
        molecule.name = str(self.source_filename)[:-4]
        molecule.charge = 0
        molecule.solvent = "thf"
        molecule.Hconnected()
        return molecule

    def molecule_xyz(
        self,
    ) -> Structure:
        """
        Helper function that returns a Structure object from an xyz file.
        No bonding information is given.

        Returns
        -------
        Structure
            Structure object
        """
        atmlst: list[Atom] = []
        with open(self.source_filename, "r") as f:
            checkln = 0
            atmnumb = 1
            for line in f.readlines():
                if checkln == 2:
                    lineproc = line.split()
                    atom = Atom(
                        lineproc[0], [float(lineproc[1]), float(lineproc[2]), float(lineproc[3])]
                    )
                    atmlst.append(atom)
                    atom.number = atmnumb
                    if atom.label == "H":  # choosing atom type
                        atom.type = "light"
                    else:
                        atom.type = "heavy"
                    atmnumb += 1
                else:
                    checkln += 1

        atmlst = recenter(atmlst)
        molecule = Structure(atmlst)
        molecule.connectivity_by_distance()
        molecule.Hconnected()
        molecule.name = str(self.source_filename)[:-4]
        molecule.charge = 0
        return molecule

    @staticmethod
    def molecule_from_rdkit(rdkit_mol: Chem.rdchem.Mol) -> Structure:
        """
        Helper function that returns a Structure object
        from an RDKIT Mol object.

        Parameters
        ----------
        rdkit_mol
        |   rdkit molecule object

        Returns
        -------
        Structure
            Structure object
        """

        molH = Chem.AddHs(rdkit_mol)
        AllChem.EmbedMolecule(molH, AllChem.ETKDG())
        mol_coordinates = []

        for i, atom in enumerate(molH.GetAtoms()):
            positions = molH.GetConformer().GetAtomPosition(i)
            atm = Atom(atom.GetSymbol(), [positions.x, positions.y, positions.z])
            mol_coordinates.append(atm)

        molecule = Structure(mol_coordinates)
        molecule.connectivity_by_distance()
        molecule.Hconnected()

        return molecule

    @staticmethod
    def molecule_from_conformer(conf: Conformer) -> "Structure":
        """
        Generate a Structure object from the conformer object of an Isomer.
        The function creates an xyz object from the conformer and
        that is transformed into a Structure object.

        Parameters
        ----------
        conf
            Conformer istance of an Isomer

        Returns
        -------
        Structure
            Structure object
        """

        # Create a xyz block from the conformer object
        atoms: list[str] = [atom.GetSymbol() for atom in list(conf.parent._molecule.GetAtoms())]
        try:
            conf.get_tag("name")
            label = str(conf.get_tag("name"))
        except KeyError:
            label = "no_name"

        xyz_buffer = io.StringIO()
        xyz_buffer.write(str(conf.parent.n_atoms) + "\n" + label + "\n")

        for j, position in enumerate(conf.coordinates):
            xyz_buffer.write(
                atoms[j]
                + " "
                + str("{0:.5f}".format(position[0]))
                + " "
                + str("{0:.5f}".format(position[1]))
                + " "
                + str("{0:.5f}".format(position[2]))
                + "\n"
            )
        xyz_block = xyz_buffer.getvalue()
        xyz_buffer.close()

        # Creating a structure object from the xyz block
        f = xyz_block.split("\n")
        checkln = 0
        atmnumb = 1
        atmlst: list[Atom] = []
        for line in f:
            if atmnumb <= conf.parent.n_atoms:
                if checkln == 2:
                    lineproc = line.split()
                    atom = Atom(
                        lineproc[0], [float(lineproc[1]), float(lineproc[2]), float(lineproc[3])]
                    )
                    atmlst.append(atom)
                    atom.number = atmnumb
                    if atom.label == "H":  # choosing atom type
                        atom.type = "light"
                    else:
                        atom.type = "heavy"
                    atmnumb += 1
                else:
                    checkln += 1

        molecule = Structure(atmlst)
        molecule.connectivity_by_distance()
        molecule.Hconnected()

        return molecule

    @staticmethod
    def molecule_from_json(json_tag: list[ConfTag], name_tag: str) -> Structure:
        """
        Generate a Structure object from the conformer geometry saved
        as a json tag of an Isomer.

        Parameters
        ----------
        json_tag
            Conformer tag of the Isomer object

        Returns
        -------
        Structure
            Structure object
        """
        

        atmlst: list[Atom] = []
        atmnumb = 1
        for atm in json_tag:
            atm_json = atm["atoms"]
            for diz in atm_json:
                coords = diz["coords"]
                atom = Atom(
                    diz["element"], [float(coords[0]), float(coords[1]), float(coords[2])]
                )
                atmlst.append(atom)
                atom.number = atmnumb
                if atom.label == "H":  # choosing atom type
                    atom.type = "light"
                else:
                    atom.type = "heavy"
                atmnumb += 1
        molecule = Structure(atmlst)
        molecule.connectivity_by_distance()
        molecule.name = name_tag
        molecule.Hconnected()

        return molecule


"""
Auxillary functions
"""


def check_identical(bond1: Bond, bond2: Bond) -> bool:
    """
    Check if two bonds are identical, by checking the identity
    of the two atoms forming the bond. Distance can be different.

    Parameters
    ---------
    bond1
        first bond of the pair to be checked
    bond2
        second bond of the pair to be checked

    Returns
    -------
    bool
        Boolean, True if both labels are the same.
    """

    return bond1.atom1.label == bond2.atom1.label and bond1.atom2.label == bond2.atom2.label


def check_connectivity(structure_m: Structure, structure_n: Structure) -> bool:
    """
    Check if two Structure objects have the same connectivity, by checking that
    they have the same number of bonds and that for each bond the labels of
    the two atoms participating are the same. Distances can be different.

    Parameters
    ---------
    structure_m
        first structure of the pair to be checked
    structure_n
        second structure of the pair to be checked

    Returns
    -------
    bool
        Boolean, True if all bond pairs are matching.
    """

    connectivity_m = structure_m.connectivity
    connectivity_n = structure_n.connectivity

    if len(connectivity_m) != len(connectivity_n):
        return False

    for bond_m, bond_n in zip(connectivity_m, connectivity_n):
        if not check_identical(bond_m, bond_n):
            return False

    return True


def is_number(s: str) -> bool:
    """

    Check if the string is a number

    Parameters
    ----------
    s
        string to check

    Returns
    -------
    bool
        Boolean regarding string
    """

    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def is_integer(s: str) -> bool:
    """

    Check if a string is an integer

    Parameters
    ----------
    bool
        string to check
    """
    if is_number(s):
        a = int(float(s))
        if a - float(s) == 0:
            return True
        else:
            return False
    else:
        return False


def strip_numbers(line: str) -> str:
    """

    Removes all numbers from a line and returns it

    Parameters
    ----------
    line
        line to process

    Returns
    -------
    str
        line stripped of numbers
    """

    line_list = list(line)
    line_letters: list[str] = []
    for j in line_list:
        if not is_number(j):
            line_letters.append(j)
    i = 0
    line_strip = ""
    while i < len(line_letters):
        line_strip += line_letters[i]
        i += 1
    return line_strip


def label_format(label: str) -> str:
    """
    Format upper and lower case for a string. Returns formatted string.

    Parameters
    ----------
    label
        string to format

    Returns
    -------
    str
        formatted string
    """
    s = list(label)
    if len(s) == 2:
        if s[1].isupper():
            s[1] = s[1].lower()
    label_new = ""
    for i in s:
        label_new += i
    return label_new


def recenter(atmlst: list[Atom]) -> list[Atom]:
    """
    Recenter atomic coordinates of a list of atoms on their center
    of mass. Returns the traslated list of Atom objects.

    Parameters
    ----------
    atmlst
        list of Atom objects to re-center.

    Returns
    -------
    list[Atom]
        re-centered atom list
    """

    xsum = 0.0
    ysum = 0.0
    zsum = 0.0
    for i in atmlst:
        xsum += i.position[0]
        ysum += i.position[1]
        zsum += i.position[2]
    xaver = xsum / len(atmlst)
    yaver = ysum / len(atmlst)
    zaver = zsum / len(atmlst)
    for j in atmlst:
        j.position[0] -= xaver
        j.position[1] -= yaver
        j.position[2] -= zaver
    return atmlst
