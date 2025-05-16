import numpy as np
import pandas as pd
import json
from pathlib import Path
import re

p = Path(__file__).parent

with open(p / "rename" / "rename_dicts.json", "r", encoding="utf-8") as f:
    rename_dicts = json.load(f)

def check_nan(value):
    return pd.isna(value)

def err2print(function, *args, **kargs):
    try:
        return function(*args, **kargs)
    except Exception as e:
        print(e)
        return np.nan

def unify_synonyms(value, type_, use_lower=True, check_nan=check_nan):
    '''use_lower: shape, activity, polymers, surfactant, activity'''
    if check_nan(value):
        return np.nan
    if use_lower:
        map_ = {key.lower(): value for key, value in rename_dicts[type_].items()}
        if value.lower() in map_:
            new_name = map_[value.lower()]
            return new_name if new_name != 0 else np.nan
        raise ValueError(f"Value '{value.lower()}' not in {type_} dict")

elems = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
         'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
         'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
         'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
         'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
         'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
         'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg',
         'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
elems.remove("I")

ignore = ['CCPK2', 'H2O2', "CDs", "CYP119", "AuNCs", "Pd1Pt1", "SBP", "WO3", "VPO1", "CPO", "WC",
          "PB", "CYP25", "BCP", "HPO", "Pd-WO3-SiO2-PB", "CF-H-Au", "CCP", "CuNCs",
          "Fe-SAs/NC", "PSH", "CH-Cu", "N-CNTs", "HCO4-", "SNC", "Fe/NC-SAs",
          "P@Pt@P", "AuNP1", "[SVW11O40]3-", "[SV2W10O40]4-", "FeP", "AgNCs",
          "WS2/AgNCs", "Fe3O4@CP", "Pt@PCN222-Mn", "W51F", "W191F", "NCNTs@MoS2",
          "AuNPTs", "Cu@PB", "PC-OOH", "Ni-FeHCF", "FeBi-NC", "SWCNHs-COOH", "K42Y",
          "K42N", "Cu23", "TiO2NTs@MoS2", "PB-Fe2O3", "Au2Pt1", "Cu-FeHCF", "Mn-FeHCF",
          "BNNS@CuS", "W2C", "Au10", "LaFeO3", "CuZnFeS", "Au-NPFe2O3NC", "MoS2/C-Au600",
          "CF", "AuPt@SF", "VO2(B)", "AuPtPd", "Bi-Au", "FeBNC", "CuHCF", "HCNTs", "V90H",
          "Pd/Fe3O4@C", "Mn3O4/Pd@Pt", "Cu2Fe(CN)6", "Os", "SeO32-", "HOP",
          "AuNP", "Y67F", "K72W", "CBP", "H39S", "H39C", "Fe3C-NC", "BP", "C1", "C2",
          "C3", "C4", "CH3Se-", "Sm-CeO2", "Co@NCNTs/NC", "N-CNTs@Co",

          "Fe-N-C", "Zn-N-C", "Co-N-C", "Au-NCs", "CS-MoSe2", "Cu-N-C", "BON", "CoPW11O39",
          "H5[PV2W10O40]", "Au-NPFe2O3","K10P2W18Fe4(H2O)2O68", "Au@SeH", "C96H160O66N2Se2",
          "C5H12NO80Se", "Co2(OH)2CO3-CeO2", "AuNF", "CPC", "PC-OH", "AuVCs", "HP", "PCB1",
          "Hf18O10(OH)26(SO4)13(H2O)33",
         ]

def is_valid_formula(formula: str) -> bool:
    if isinstance(formula, float):
        return False
    for i in ignore:
        if i in formula:
            return False
    return True if re.fullmatch(rf"(?:\(?\[?(?:\(?(?:{'|'.join(elems)})+\)?\d*)+\]?\)?\d*[@\-/]?)+", formula) else False