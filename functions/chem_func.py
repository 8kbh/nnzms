import functions_a as fu
import unifyer as uf
import numpy as np


h_prop = {}
h_smiles = {}
def get_gescriptors_by_names(names,
                             pubchem_prop=['MolecularWeight', 'XLogP','TPSA', 'Complexity'],
                             rdkit_prop=["MolWt", "PEOE_VSA7", "PEOE_VSA9", "VSA_EState8", "Kappa2",
                                         "BalabanJ", "MinAbsEStateIndex", "MinEStateIndex", "EState_VSA6",
                                         "VSA_EState4", "PEOE_VSA8", "MinPartialCharge", "EState_VSA4",
                                         "SMR_VSA7", "BCUT2D_CHGLO", "MaxEStateIndex", "MaxPartialCharge"],
                             use_history=True, verbose=True):
    '''
    names - List of names of chemicals
    '''

    ### pubchem descriptors
    # mw = 0
    tpsa = 0
    comp = 0
    logp = 0 if len(names) == 1 else np.nan
    for name in names:
        if name not in h_prop:
            h_prop[name] = {}
            prop_obj = fu.pcp.get_properties(
                properties_list,
                fu.monomer(name), 
                'name'
            )
            if len(prop_obj) < 1:
                # if molecue not found then set nan values
                if verbose:
                    print(f"ERR PCP not found get_properties {name}")
                # mw = np.nan
                tpsa = np.nan
                P = np.nan
                comp = np.nan
                break

            for property_ in porperties:
                h_prop[name][property_] = prop_obj[0].get(property_, None)
            

        # polym = float(mcoat)*1000/float(mw) ## i don't have mcoat
        prop_dict = h_prop[name]
        if len(names) == 1:
            logp = prop_dict.get('XLogP', np.nan)
        # mw += float(prop_dict.get('MolecularWeight', np.nan))
        tpsa += float(prop_dict.get('TPSA', np.nan))
        comp += float(prop_dict.get('Complexity', np.nan))

    # convert zeros to nan
    # mw = mw if mw else np.nan
    if tpsa == 0 or logp == 0 or comp == 0:
        print(names, tpsa, logp, comp)
    # tpsa = tpsa if tpsa else np.nan
    # logp = logp if logp else np.nan
    # comp = comp if comp else np.nan

    desc_pubchem = {
        # "MolWt": mw,
        "XLogP": logp,
        "TPSA": tpsa,
        "Complexity": comp,
    }

    ### RDKit descriptors
    smiles_list = []
    for name in names:
        if name not in h_smiles:
            smiles_obj = fu.pcp.get_compounds(fu.monomer(name), 'name')

            if len(smiles_obj) < 1:
                h_smiles[name] = ""
                if verbose:
                    print(f"ERR PCP not found get_compounds {name}")
            else:
                h_smiles[name] = smiles_obj[0].isomeric_smiles
        smiles_list.extend(h_smiles[name])
        

    if "" in smiles_list:
        desc_rdkit = {key: np.nan for key in rdkit_prop}
    else:
        mol = fu.Chem.MolFromSmiles(".".join(smiles_list))
        # print(smiles_list)
        allDescrs = fu.getMolDescriptors(mol)
        # print(allDescrs.keys())
        desc_rdkit = {key: allDescrs[key] for key in rdkit_prop}

    desc = {**desc_pubchem, **desc_rdkit}
    return desc


def get_descriptors(*chemicals):
    rd = {}
    
    ### descriptors of surface, pol, surf
    chemicals = [ch for ch in chemicals if str(ch) not in ["0", "nan", "naked"] and not uf.check_nan(ch)]
    # print(chemicals)

    chemicals_descriptors = get_gescriptors_by_names(chemicals)
    
    # remove descriptors that aren't calculated for (surface, pol, surf)
    chemicals_descriptors = {key: value for key, value in chemicals_descriptors.items() if key not in ["MaxEStateIndex", "MaxPartialCharge", "Complexity"]}
    # print(chemicals_descriptors)
    rd.update(chemicals_descriptors)

    ### descriptors of reaction_type (ex.: TMB + H2O2)
    rd[f"MinPartialCharge.1"] = np.nan
    rd[f"MaxPartialCharge.1"] = np.nan
    rd[f"Complexity1"] = np.nan
    for i in range(2):
        rd[f"TPSA{i+1}"] = np.nan
        rd[f"MaxEStateIndex.{i+1}"] = np.nan


    if uf.check_nan(row["ReactionType"]):
        return rd
        

    chems = row["ReactionType"].replace(" + ", "+").split("+")
    chems = [ch.lstrip().rstrip() for ch in chems if ch not in ["0", "nan", ""] and not uf.check_nan(ch)]
    if len(chems) > 2:
        print("A lot of elements in reaction: ", "id", row["ReactionType"], chems)
        return rd

    for i, chem in enumerate(chems):
        desc = get_gescriptors_by_names((chem,))
        rd[f"TPSA{i+1}"] = desc["TPSA"]
        rd[f"MaxEStateIndex.{i+1}"] = desc["MaxEStateIndex"]
        if i == 0:
            rd[f"Complexity{i+1}"] = desc["Complexity"]
            rd[f"MinPartialCharge.{i+1}"] = desc["MinPartialCharge"]
            rd[f"MaxPartialCharge.{i+1}"] = desc["MaxPartialCharge"]

    return rd