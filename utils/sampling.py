from rdkit import Chem, Geometry
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from .featurization import featurize_mol_from_smiles
import functools
import copy
import torch
import numpy as np
from .graph import get_all_simple_4_path

def smi_to_pyg(smi, dataset='drugs'):
    '''
    Convert smile to graph
    Get edge_mask, mask_rotate
    '''    
    mol, data = featurize_mol_from_smiles(smi, dataset=dataset) # Return graph from smile
    if not mol:
        return None, None
    return mol, data

def sample(args, mol, data, model, n_confs, steps=500, batch_size=32):
    '''
    Copy from ConformerDataset.dataset.py
    data : Data(x=[19, 44], edge_index=[2, 36], edge_attr=[36, 6], z=[19], name='C#CC#C[C@@H](CC)CO')
    conformers List([Data]):
    steps: Number of inference steps used by the resampler
    Return:
        List([Data])
    '''
    # Get all simple 4 paths
    simple_4_paths = get_all_simple_4_path(data) # Shape: (P,4)
    num_paths = len(simple_4_paths)
    if not num_paths:
        return None
    simple_4_paths = [list(i) for i in zip(*(x for x in simple_4_paths))] # Shape: (4,P)
    data.simple_4_paths = torch.tensor(simple_4_paths).T
    data.num_paths = num_paths

    # Loader
    conformers = [copy.deepcopy(data).to(args.device) for i in range(n_confs)]
    conf_dataset = InferenceDataset(conformers)
    loader = DataLoader(conf_dataset, batch_size=batch_size, shuffle=False)
    
    # Sampling utilities
    sampler = Euler_Maruyama_sampler
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=args.sigma)

    # Batch Sampling
    directions = []
    for data in loader:
        # Encode

        # Embed node attribute
        node_attr, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        data.node_attr = model.embedding(node_attr, edge_index, edge_attr)
        directions.append(sampler(model.score_model, data,
                        marginal_prob_std_fn,
                        diffusion_coeff_fn,  
                        device=args.device))

    # Convert to 3D coordinates
    sphe_pos = torch.cat(sphe_pos, dim=0)
    # sphe_pos[:,3] = (sphe_pos[:,3]*3.5).clamp(0.1, 4.0)
    # sphe_pos[:,:3] = (sphe_pos[:,:3]*2-1).clamp(-1.0, 1.0)
    print(sphe_pos)
    print(sphe_pos.shape)
    raise



    # Distribute sphe_pos
    sphe_pos = torch.cat(sphe_pos, dim=0)
    num_nodes = data.x.shape[0]
    for i in range(n_confs):
        conformers[i].sphe_pos = sphe_pos[i*num_nodes:(i+1)*num_nodes]

    return conformers

def Euler_Maruyama_sampler(score_model, data,
                             marginal_prob_std,
                             diffusion_coeff, 
                             num_steps=500, 
                             device='cuda', 
                             eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        eps: The smallest time step for numerical stability.
    
    Returns:
        Samples.    
    """
    # Number of nodes
    num_nodes = data.x.shape[0]
    
    # Initial T (1) for each batch
    t = torch.ones(len(data.ptr)-1, device=device)

    # Sampling initial directions with T = 1
    std = marginal_prob_std(t)
    std = torch.cat([std[i].tile((num_path,1)) for i, num_path in enumerate(data.num_paths)], dim=0)
    directions = torch.randn(data.num_paths.sum().item(), 3, device=device) \
                    * std
    
    # Time steps for reverse process
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    
    x = directions
    with torch.no_grad():
        loss_tot = []
        for time_step in time_steps:      
            batch_time_step = torch.ones(len(data.ptr)-1, device=device) * time_step # Shape: (Num_nodes)
            score = score_model(data, batch_time_step, x)
            g = diffusion_coeff(time_step)
            
            # Formula
            mean_x = x + (g**2) * score * step_size
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)

            # Report
            z = torch.randn_like(x)
            std = marginal_prob_std(torch.tensor([time_step], device='cuda'))
            loss_tot += [torch.mean(torch.sum((score * std ), dim=1)).item()]
        print(sum(loss_tot)/len(loss_tot))
    print(mean_x)
    # Do not include any noise in the last sampling step.
    return mean_x

class InferenceDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        for i, d in enumerate(data_list):
            d.idx = i
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]

def marginal_prob_std(t, sigma):
    r"""Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
    
    Returns:
        The standard deviation.
    """
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))

def diffusion_coeff(t, sigma, device='cuda'):
    r"""Compute the diffusion coefficient of our SDE.

    Args:
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.
    
    Returns:
        The vector of diffusion coefficients.
    """
    return sigma**t.to(device)

def pyg_to_mol(mol, data, mmff=False, rmsd=True, copy=True):
    if not mol.GetNumConformers(): # If there is no confs.
        conformer = Chem.Conformer(mol.GetNumAtoms()) # The class to store 2D or 3D conformation of a molecule
        mol.AddConformer(conformer)
    coords = data.pos
    if type(coords) is not np.ndarray:
        coords = coords.double().numpy()

    # Transfer Atom Position from $data to $mol
    for i in range(coords.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
    if mmff:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, mmffVariant='MMFF94s') # Uses MMFF to optimize all of a molecule's conformations
        except Exception as e:
            pass

    if not copy: return mol
    return deepcopy(mol)