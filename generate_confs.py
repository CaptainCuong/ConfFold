from argparse import ArgumentParser
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import pickle
import pandas as pd
from tqdm import tqdm
import yaml
import os.path as osp
from utils.utils import get_model
import torch
from utils.sampling import smi_to_pyg, sample

parser = ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./test_run', help='Path to folder with trained model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
parser.add_argument('--out', type=str, default='test_run/test_data.pkl', help='Path to the output pickle file')
parser.add_argument('--test_csv', type=str, default='./data/QM9/test_smiles.csv', help='Path to csv file with list of smiles and number conformers')
parser.add_argument('--inference_steps', type=int, default=500, help='Number of denoising steps')
parser.add_argument('--limit_mols', type=int, default=300, help='Limit to the number of molecules')
parser.add_argument('--confs_per_mol', type=int, default=None, help='If set for every molecule this number of conformers is generated, '
                                                                    'otherwise 2x the number in the csv file')
# parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE instead of the SDE')
parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')
parser.add_argument('--sampling_steps', type=int, default=500, help='Number of conformers generated in parallel')
parser.add_argument('--batch_size', type=int, default=32, help='Number of conformers generated in parallel')

args = parser.parse_args()

"""
    Generates conformers for a list of molecules' SMILE given a trained model
    Saves a pickle with dictionary with the SMILE as key and the RDKit molecules with generated conformers as value 
"""


##################### Load model
with open(f'{args.model_dir}/model_parameters.yml') as f:
    args.__dict__.update(yaml.full_load(f))

model = get_model(args)
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

##################### Load generated smi
test_data = pd.read_csv(args.test_csv).values
if args.limit_mols:
    test_data = test_data[:args.limit_mols]


conformer_dict = {}
test_data = tqdm(enumerate(test_data), total=len(test_data))
         
#######################################

def sample_confs(n_confs, smi):
    '''
    '''
    mol, data = smi_to_pyg(smi, dataset=args.dataset)
    conformers = sample(args, mol, data, model, n_confs, steps=args.sampling_steps, batch_size=32)
    
    if not conformers:
        raise
        return []
    mols = [pyg_to_mol(mol, conf) for conf in conformers] # Convert pyg object to Mol object
    return mols


for smi_idx, (n_confs, smi) in test_data:
    if type(args.confs_per_mol) is int:
        mols = sample_confs(args.confs_per_mol, smi)
    else:
        mols = sample_confs(2 * n_confs, smi)
    if not mols: continue
    if not args.no_energy:
        rmsd = [mol.rmsd for mol in mols]
        dlogp = np.array([mol.euclidean_dlogp for mol in mols])
        if args.xtb:
            energy = np.array([mol.xtb_energy for mol in mols])
        else:
            energy = np.array([mol.mmff_energy for mol in mols])
        F, F_std = (0, 0) if args.no_energy else free_energy(dlogp, energy)
        print(
            f'{smi_idx} rotable_bonds={mols[0].n_rotable_bonds} n_confs={len(rmsd)}',
            f'rmsd={np.mean(rmsd):.2f}',
            f'F={F:.2f}+/-{F_std:.2f}',
            f'energy {np.mean(energy):.2f}+/-{bootstrap((energy,), np.mean).standard_error:.2f}',
            f'dlogp {np.mean(dlogp):.2f}+/-{bootstrap((dlogp,), np.mean).standard_error:.2f}',
            smi,
            flush=True
        )
    else:
        print(f'{smi_idx} rotable_bonds={mols[0].n_rotable_bonds} n_confs={len(mols)}', smi, flush=True)
    conformer_dict[smi] = mols

# save to file
if args.out:
    with open(f'{args.out}', 'wb') as f:
        pickle.dump(conformer_dict, f)
print('Generated conformers for', len(conformer_dict), 'molecules')
