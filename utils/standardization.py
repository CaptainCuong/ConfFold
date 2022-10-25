import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

def sort_confs(confs):
    return sorted(confs, key=lambda conf: -conf['boltzmannweight'])

def resample_confs(confs, max_confs=None):
    weights = [conf['boltzmannweight'] for conf in confs]
    weights = np.array(weights) / sum(weights)
    k = min(max_confs, len(confs)) if max_confs else len(confs)
    return random.choices(confs, weights, k=k)

def clean_confs(smi, confs, limit=None):
    good_ids = []
    try:
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False) # standard smi (because smiles may contains hydrogen)
    except Exception as e:
        print('Error', smi, e)
        return []
    for i, c in enumerate(confs):
        conf_smi = Chem.MolToSmiles(Chem.RemoveHs(c['rd_mol'], sanitize=False),
                                    isomericSmiles=False) # used to compare with the standard smi
        if conf_smi == smi:
            good_ids.append(i)
        if len(good_ids) == limit:
            break
    return [confs[i] for i in good_ids]