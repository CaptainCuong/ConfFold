import glob, pickle, tqdm
from collections import defaultdict
from argparse import ArgumentParser
from utils.standardization import *
from scipy.optimize import linear_sum_assignment
import datamol as dm
import networkx as nx

parser = ArgumentParser()
parser.add_argument('--worker_id', type=int, default=1, help='Worker id to determine correct portion')
parser.add_argument('--out_dir', type=str, default='data/QM9/standardized_pickles', help='Output directory for the pickles')
parser.add_argument('--jobs_per_worker', type=int, default=100, help='Number of molecules for each worker')
parser.add_argument('--root', type=str, default='data/QM9/qm9/', help='Directory with molecules pickle files')
parser.add_argument('--popsize', type=int, default=15, help='Population size for differential evolution')
parser.add_argument('--max_iter', type=int, default=15, help='Maximum number of iterations for differential evolution')
parser.add_argument('--confs_per_mol', type=int, default=30, help='Maximum number of conformers to take for each molecule')
parser.add_argument('--mmff', action='store_true', default=True, help='Whether to relax seed conformers with MMFF before matching')
parser.add_argument('--no_match', action='store_true', default=False, help='Whether to skip conformer matching')
parser.add_argument('--boltzmann', choices=['top', 'resample'], default=None, help='If set, specifies a different conformer selection policy')
args = parser.parse_args()

root = args.root
files = sorted(glob.glob(f'{root}*.pickle'))
aro_f = None
confs = None
mol = None
writer = open('log.csv', 'w')
for f in tqdm(files):
	if not os.path.exists(f'visualization/{key}'):
		os.makedirs(f'visualize/{key}')
    with open(f, 'rb') as pkl:
        mol_dic = pickle.load(pkl)
    confs = mol_dic['conformers']
    name = mol_dic['smiles']
    mol = confs[0]['rd_mol']
    if dm.descriptors.n_aromatic_atoms(mol) > 0:
        aro_f = f
        break

