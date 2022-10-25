import os.path
from multiprocessing import Pool

from rdkit import Chem
import numpy as np
import glob, pickle, random
import os.path as osp
import torch, tqdm
import copy
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from collections import defaultdict
from utils.graph import tree_converter_dfs, tree_converter_bfs, get_all_simple_4_path
from utils.featurization import dihedral_pattern, featurize_mol, qm9_types, drugs_types
from .geometry import get_direction
import numpy as np

class PreprocessTransform(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        '''
        Data(x=[19, 44], edge_index=[2, 40], edge_attr=[40, 4], z=[19], canonical_smi='C#CCC12CCC1(O)C2', 
        mol=<rdkit.Chem.rdchem.Mol object at 0x000001E033C71780>, pos=[8], weights=[8], 
        simple_4_paths=[4, 134], num_paths=134, directions=[8])
        '''
        # randomly select positions of A conformer
        i = random.choice(range(len(data.pos)))
        data.pos = data.pos[i]
        data.directions = data.directions[i]

        return data

class ConformerDataset(Dataset):
    def __init__(self, root, split_path, mode, types, dataset, transform=None, num_workers=1, limit_molecules=None,
                 cache=None, pickle_dir=None):
        # part of the featurisation and filtering code taken from GeoMol https://github.com/PattanaikL/GeoMol
        '''
        Args:
            root:. Example: (str) 'data/QM9/qm9/'
            split_path:. Example: (str) 'data/QM9/split.npy'
            mode:. Example: (str) 'train'
            types:. Example: (dict) {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            dataset:. Example: (str) 'qm9'
            transform:. Example: TorsionNoiseTransform(sigma_min=0.001, sigma_max=2)
            cache:. Example: (str) 'data/QM9/cache'
            pickle_dir:. Example: (str) 'data/QM9/qm/'
        '''
        super(ConformerDataset, self).__init__(root, transform)
        self.root = root
        self.types = types
        self.failures = defaultdict(int)
        self.dataset = dataset

        if cache: cache += "." + mode
        self.cache = cache
        if cache and os.path.exists(cache):
            print('Reusing preprocessing from cache', cache)
            with open(cache, "rb") as f:
                self.datapoints = pickle.load(f)
        else:
            print("Preprocessing")
            self.datapoints = self.preprocess_datapoints(root, split_path, pickle_dir, mode, num_workers, limit_molecules)
            if cache:
                print("Caching at", cache)
                with open(cache, "wb") as f:
                    pickle.dump(self.datapoints, f)

        if limit_molecules:
            self.datapoints = self.datapoints[:limit_molecules]


    def preprocess_datapoints(self, root, split_path, pickle_dir, mode, num_workers, limit_molecules):
        '''
        Args:
            pickle_dir: Folder in which the pickle are put after standardisation/matching
            root: data/QM9/qm9/
            split_path: . Default value is: (str) 'data/QM9/split.npy'
            pickle_dir: . Default value is: (str) 'data/QM9/qm/'
            mode: Used to set the name of cache file: Example value is: (str) 'train'
            num_workers: 1
            limit_molecules: Limit to the number of molecules in dataset, 0 uses them all.
        '''
        mols_per_pickle = 1000
        split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
        split = sorted(np.load(split_path, allow_pickle=True)[split_idx])
        if limit_molecules:
            split = split[:limit_molecules]
        if self.dataset == 'qm9':
            split = [i for i in split if i < 133111]
        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))
        smiles = smiles[split] # Path to pickle

        self.open_pickles = {}
        if pickle_dir:
            smiles = [(i // mols_per_pickle, smi[len(root):-7]) for i, smi in zip(split, smiles)]
            # smiles: (file number, smile) 
            if limit_molecules:
                smiles = smiles[:limit_molecules]
            self.current_pickle = (None, None)
            self.pickle_dir = pickle_dir
        else:
            smiles = [smi[len(root):-7] for smi in smiles]

        print('Preparing to process', len(smiles), 'smiles')
        datapoints = []
        if num_workers > 1:
            p = Pool(num_workers)
            p.__enter__()
        with tqdm.tqdm(total=len(smiles)) as pbar:
            map_fn = p.imap if num_workers > 1 else map
            for t in map_fn(self.filter_smiles, smiles):
                if t:
                    datapoints.append(t)
                pbar.update() # Print update with tqdm
        if num_workers > 1: p.__exit__(None, None, None)
        print('Fetched', len(datapoints), 'mols successfully')
        print(self.failures)
        if pickle_dir: del self.current_pickle
        return datapoints

    def filter_smiles(self, smile):
        '''
        Read from standardize path Ex: 'data/QM9/standardized_pickles/000.pickle'
        Filter smiles that cannot be handled
        Featurize and convert to `torch_geometric.data.Data`
        Add tree
        Compute feature for each edge [torsional_angle,bond_angle,length]
        Args:
            smile: (pickle_file_id,smile)
        '''
        temp_smile = smile
        if type(smile) is tuple:
            pickle_id, smile = smile
            current_id, current_pickle = self.current_pickle # (pickle_file_id, pickle_data)
            if current_id != pickle_id:
                path = osp.join(self.pickle_dir, str(pickle_id).zfill(3) + '.pickle')
                # Check if 'standardize' path exists. Ex: 'data/QM9/standardized_pickles/000.pickle'
                if not osp.exists(path):
                    self.failures[f'std_pickle{pickle_id}_not_found'] += 1
                    return False
                with open(path, 'rb') as f:
                    self.current_pickle = current_id, current_pickle = pickle_id, pickle.load(f)
            if smile not in current_pickle:
                self.failures['smile_not_in_std_pickle'] += 1
                return False
            mol_dic = current_pickle[smile]

        else:
            if not os.path.exists(os.path.join(self.root, smile + '.pickle')): # Not in 'data/QM9/qm9/'
                self.failures['raw_pickle_not_found'] += 1
                return False
            pickle_file = osp.join(self.root, smile + '.pickle')
            mol_dic = self.open_pickle(pickle_file)
        
        smile = mol_dic['smiles']

        if '.' in smile: # Disconnected structures
            self.failures['dot_in_smile'] += 1
            return False

        # filter mols rdkit can't intrinsically handle
        mol = Chem.MolFromSmiles(smile)
        if not mol:
            self.failures['mol_from_smiles_failed'] += 1
            return False
        try:
            mol = mol_dic['conformers'][0]['rd_mol']
        except:
            print(len(mol_dic['conformers']))
            print(temp_smile)
            raise
        N = mol.GetNumAtoms()
        if not mol.HasSubstructMatch(dihedral_pattern):
            self.failures['no_substruct_match'] += 1
            return False

        if N < 4:
            self.failures['mol_too_small'] += 1
            return False

        data = self.featurize_mol(mol_dic)
        if not data:
            self.failures['featurize_mol_failed'] += 1
            return False
        
        return data

    def len(self):
        return len(self.datapoints)

    def get(self, idx):
        data = self.datapoints[idx]
        return copy.deepcopy(data)

    def open_pickle(self, mol_path):
        with open(mol_path, "rb") as f:
            dic = pickle.load(f)
        return dic

    def featurize_mol(self, mol_dic):
        '''
        Convert Mol object into features
        Args:
            mol_dic: dict containing keys ['conformers', 'totalconfs', 'temperature', 
                'uniqueconfs', 'lowestenergy', 'poplowestpct', 
                'ensembleenergy', 'ensembleentropy', 'ensemblefreeenergy', 
                'charge', 'smiles']
        Returns
            (torch_geometric.data.data.Data) with keys:
            ['x', 'edge_index', 'edge_attr', 'z', 'canonical_smi', 'mol', 'pos', 'weights']
        '''
        confs = mol_dic['conformers']
        name = mol_dic["smiles"]

        mol_ = Chem.MolFromSmiles(name)
        canonical_smi = Chem.MolToSmiles(mol_, isomericSmiles=False)

        pos = []
        weights = []
        for conf in confs:
            mol = conf['rd_mol']

            # filter for conformers that may have reacted
            try:
                conf_canonical_smi = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False), isomericSmiles=False)
            except Exception as e:
                print(e)
                continue

            if conf_canonical_smi != canonical_smi:
                continue

            pos.append(torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float))
            weights.append(conf['boltzmannweight'])
            correct_mol = mol

        # return None if no non-reactive conformers were found
        if len(pos) == 0:
            return None

        data = featurize_mol(correct_mol, self.types)
        normalized_weights = list(np.array(weights) / np.sum(weights))
        if np.isnan(normalized_weights).sum() != 0:
            print(name, len(confs), len(pos), weights)
            normalized_weights = [1 / len(weights)] * len(weights)

        data.canonical_smi, data.mol, data.pos, data.weights = canonical_smi, correct_mol, pos, normalized_weights
        
        # Get all simple 4 paths
        simple_4_paths = get_all_simple_4_path(data) # Shape: (P,4)
        num_paths = len(simple_4_paths)
        if not num_paths:
            return None
        # simple_4_paths = [list(i) for i in zip(*(x for x in simple_4_paths))] # Shape: (4,P)
        simple_4_paths = torch.tensor(simple_4_paths)
        data.simple_4_paths = simple_4_paths
        assert data.simple_4_paths.shape[1] == 4
        data.num_paths = num_paths

        # Add directions for each pos
        directions = []
        for pos in data.pos:
            try:
                direction = get_direction(pos, simple_4_paths) # Shape: (D, 3)
            except:
                return None
            directions.append(torch.tensor(direction))
        data.directions = directions

        return data

    def resample_all(self, resampler, temperature=None):
        ess = []
        for data in tqdm.tqdm(self.datapoints):
            ess.append(resampler.resample(data, temperature=temperature))
        return ess


def construct_loader(args, modes=('train', 'val')):
    if isinstance(modes, str):
        modes = [modes]

    loaders = []
    transform = PreprocessTransform()
    types = qm9_types if args.dataset == 'qm9' else drugs_types

    for mode in modes:
        dataset = ConformerDataset(args.data_dir, args.split_path, mode, dataset=args.dataset,
                                   types=types, transform=transform,
                                   num_workers=args.num_workers,
                                   limit_molecules=args.limit_train_mols,
                                   cache=args.cache,
                                   pickle_dir=args.std_pickles)

        loader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=False if mode == 'test' else True,
                            pin_memory=True)
        loaders.append(loader)

    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders