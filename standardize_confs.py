import glob, os, pickle, random, tqdm
from collections import defaultdict
from argparse import ArgumentParser
from utils.standardization import *
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

parser = ArgumentParser()
parser.add_argument('--nu_portion', type=int, default=134, help='Number of portions, 134 for QM9')
parser.add_argument('--out_dir', type=str, default='data/QM9/standardized_pickles', help='Output directory for the pickles')
parser.add_argument('--jobs_per_worker', type=int, default=1000, help='Number of molecules for each worker')
parser.add_argument('--root', type=str, default='data/QM9/qm9/', help='Directory with molecules pickle files')
parser.add_argument('--confs_per_mol', type=int, default=30, help='Maximum number of conformers to take for each molecule')
parser.add_argument('--boltzmann', choices=['top', 'resample'], default=None, help='If set, specifies a different conformer selection policy')
args = parser.parse_args()

root = args.root

for worker_id in range(args.nu_portion):
    files = sorted(glob.glob(f'{root}*.pickle'))
    files = files[worker_id * args.jobs_per_worker:(worker_id + 1) * args.jobs_per_worker]
    master_dict = {}
    print(f'{len(files)} jobs, file_id = {worker_id}')

    for i, f in enumerate(files):
        # f (str): path to pickle file
        with open(f, "rb") as pkl:
            mol_dic = pickle.load(pkl)
        confs = mol_dic['conformers'] # A list of conformers
        name = mol_dic["smiles"]

        if args.boltzmann == 'top': # sort based on boltzmann weight
            confs = sort_confs(confs)

        limit = args.confs_per_mol if args.boltzmann != 'resample' else None
        confs = clean_confs(name, confs, limit=limit)

        if not confs:
            print(f"No clean confs: smi={name}")
            continue

        if args.boltzmann == 'resample': # resample based on boltzmann weight
            confs = resample_confs(confs, max_confs=args.confs_per_mol)
        
        if args.confs_per_mol:
            confs = confs[:args.confs_per_mol]
        
        mol_dic['conformers'] = confs # Update list of conformers
        master_dict[f[len(root):-7]] = mol_dic # f[len(root):-7]: smile name

    print(f'Success molecules={len(master_dict)}/{len(files)}')
    print('-'*70)
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    with open(args.out_dir + '/' + str(worker_id).zfill(3) + '.pickle', 'wb') as f:
        pickle.dump(master_dict, f)
