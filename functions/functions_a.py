from pymatgen.core.composition import Composition, Element
import pubchempy as pcp
from rdkit.Chem import Descriptors
from rdkit import RDLogger, Chem

def CoreShell():
    class CoreShell(object):
        parts = formula.split("@")
        if "@" in parts[0]:
            comp1 = parts[0]
            comp1 = comp1.replace("-", "")
            comp2 = None
        else:
            comp1 = parts[1]
            comp1 = comp1.replace("-", "")
            comp2 = None
    return CoreShell


def Composite():
    class Composite(object):
        components = formula.split("/")
        comp1 = components[0].replace('-', '')
        comp2 = components[1].replace('-', '')
    return Composite


def Alloy():
    class Alloy(object):
        comp = formula.split("-")
        comp1 = comp[0]
        comp2 = comp[1]
    return Alloy


def Matter():
    class Matter(object):
        comp1 = formula
        comp2 = None
    return Matter


def comp_norm(formula):
    if "@" in formula:
        composition = CoreShell()
    elif "/" in formula:
        composition = Composite()
    elif "-" in formula:
        composition = Alloy()
    else:
        composition = Matter()
    return composition


def intindex(formula):
    '''Get rid of float indexes in formulas'''
    comp = Composition(formula)
    formula = comp.formula
    pattern = r'([A-Za-z]+)(\d+\.\d+|\d+)([A-Za-z]*)'
    floating_point_indices = re.findall(r'\d+\.\d+', formula)
    if len(floating_point_indices) > 0:
        def multiply(match_obj):
            index = match_obj.group(2)
            new_index = str(int(float(index) * 100))
            return match_obj.group(1) + new_index + match_obj.group(3)
        new_formula = re.sub(pattern, multiply, formula)
        return new_formula
    else:
        return formula


import csv
import os

redox1_file_path = os.path.join(os.path.dirname(__file__), 'redox1.csv')
def Redox(element, os1, os2=0):
    with open(redox1_file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip the header row
        for row in reader:
            try:
                el1, el2, potential = row
                if el1 == f"{element}[{os1}]" and el2 == f"{element}[{os2}]":
                    return float(potential)
            except ValueError:
                print(f"Error on line {row}")
    return np.nan


def OS(comp):
    comp1 = Composition(intindex(comp))
    try:
        os = comp1.oxi_state_guesses()[0]
    except IndexError:
        os = {}
        formula = comp1.formula
        pattern = r'([A-Z][a-z]*)(\d*)'
        matches = re.findall(pattern, formula)
        for i, match in enumerate(matches):
            symbol, index = match
            element_var_name = symbol
            os[element_var_name] = 0.0
    return os


import re
def elfromcomp(comp):
    '''
    split Composition to elements and their counts.
    Ex.: for Composition("H2O"), the function will return {'H': 2.0, 'O': 1.0}
    '''
    elements = {}
    formula = comp.formula
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    for i, match in enumerate(matches):
        symbol, index = match
        if index == '':
            index = '1'
        element_var_name = symbol
        elements[element_var_name] = float(index)
    return elements
    
import re
def monomer(string):
    substrings = []
    if string == "Tetrakis(4-carboxyphenyl)porphine":
        substrings.append(string)
    match = re.search(r'\((.*?)\)', string)
    if match:
        substrings.append(match.group(1))
    else:
        substrings.append(string)
    return substrings[0]


# def smiles(name):
#     smiles_list = []
#     if not name:
#         return ""

#     if ',' not in name:  # Single smiles
#         smiles = pcp.get_compounds(monomer(name), 'name')[0].isomeric_smiles
#         smiles_list.append(smiles)
#     else:  # String of molecule names
#         names = name.split(',')
#         merged_names = []
#         temp_name = ""
#         for n in names:
#             if temp_name == "":
#                 temp_name = n
#             elif temp_name.endswith('-') or temp_name.endswith('_'):
#                 temp_name += n
#             else:
#                 merged_names.append(temp_name)
#                 temp_name = n
#         merged_names.append(temp_name)

#         for merged_name in merged_names:
#             smiles = pcp.get_compounds(monomer(merged_name), 'name')[0].isomeric_smiles
#             smiles_list.append(smiles)
#     return '.'.join(smiles_list)
    

def smiles(names_list):
    smiles_list = []
    if len(names_list) == 0:
        return ""

    if len(names_list) == 1:  # Single smiles
        smiles = pcp.get_compounds(monomer(names_list[0]), 'name')[0].isomeric_smiles
        smiles_list.append(smiles)

    else:  # String of molecule names
        merged_names = []
        temp_name = ""
        for name in names_list:
            if temp_name == "":
                temp_name = name
            elif temp_name.endswith('-') or temp_name.endswith('_'):
                temp_name += name
            else:
                merged_names.append(temp_name)
                temp_name = ""
        merged_names.append(temp_name)

        for merged_name in merged_names:
            smiles = pcp.get_compounds(monomer(merged_name), 'name')[0].isomeric_smiles
            smiles_list.append(smiles)

    return smiles_list


def getMolDescriptors(mol, missingVal=None):
    res = {}
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except Exception:
            val = missingVal
        res[nm] = val
    return res


def check(mol):
    if mol == 0:
        mol = 'poly(N-Vinylpyrrolidone)'
    elif mol == 1:
        mol = 'citric acid'
    elif mol == 2:
        mol = 'ascorbic acid'
    elif mol == 3:
        mol = 'oleic acid'
    else:
        mol = 'poly(ethylene oxide)'
    return mol


def activities(number):
    if number == 1:
        activ = 'peroxidase'
    elif number == 2:
        activ = 'oxidase'
    elif number == 3:
        activ = 'catalase'
    return activ


def split(reaction):
    # Split the reaction_string using '+' as the delimiter
    pattern = r'\s*\+\s*'
    substrings = re.split(pattern, reaction)
    # Assign sub1 and sub2 accordingly
    sub1 = substrings[0]
    sub2 = substrings[1] if len(substrings) == 2 else None
    # Return sub1 and sub2
    return sub1, sub2


def systems(number):
    if number == 0:
        crystal = 'amorphous'
    elif number == 1:
        crystal = 'triclinic'
    elif number == 2:
        crystal = 'monoclinic'
    elif number == 3:
        crystal = 'orthorhombic'
    elif number == 4:
        crystal = 'tetragonal'
    elif number == 5:
        crystal = 'trigonal'
    elif number == 6:
        crystal = 'hexagonal'
    elif number == 7:
        crystal = 'cubic'
    return crystal


import pandas as pd
import numpy as np


def unlog(df, columns):
    for col in columns:
      if col =='Cconst':
        df[col] = ((df[col])**2).astype(float)
      else:
        df[col] = 10**(df[col]).astype(float)
    return df


def columns_rep(df9):
    db = pd.read_csv('/var/www/aicidlab/dizyme2/prediction/med_perox.csv')
    df9['ph'] = db['ph']
    df9['temp'] = db['temp']
    df9['lgCmin'] = db['lgCmin']
    df9['lgCmax'] = db['lgCmax']
    df9['lgCcat'] = db['lgCcat']
    df9['Cconst'] = db['Cconst'] if (df9['Cconst'] != 0).all() else 0
    df9 = df9.drop(columns=['reaction'])
    return df9