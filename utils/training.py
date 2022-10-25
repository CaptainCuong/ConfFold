import numpy as np
from tqdm import tqdm
import torch
import functools

def train_epoch(args, model, loader, optimizer, device):
    model.train()
    loss_tot = 0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)

    for data in tqdm(loader, total=len(loader)):
        
        data = data.to(device)
        optimizer.zero_grad()
        x = data.directions

        # Sampling random t for training
        random_t = torch.rand(len(data.ptr)-1, device=x.device) * (1. - args.eps) + args.eps
        
        # Perturb noise
        z = torch.randn_like(x)
        std = marginal_prob_std_fn(random_t)
        std = torch.cat([std[i].tile((num_path,1)) for i, num_path in enumerate(data.num_paths)], dim=0)
        perturbed_x = x + z * std
        
        # Score
        score = model(data, random_t, perturbed_x)
        
        # Loss function
        loss = torch.mean(torch.sum((score * std + z)**2, dim=1)) # 4.x
        loss_tot += loss.item()

    loss_avg = loss_tot / len(loader)
    return loss_avg


@torch.no_grad()
def test_epoch(args, model, loader, device):
    model.eval()
    loss_tot = 0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=args.sigma)

    for data in tqdm(loader, total=len(loader)):
        
        data = data.to(device)
        x = data.directions

        # Sampling random t for training
        random_t = torch.rand(len(data.ptr)-1, device=x.device) * (1. - args.eps) + args.eps
        
        # Perturb noise
        z = torch.randn_like(x)
        std = marginal_prob_std_fn(random_t)
        std = torch.cat([std[i].tile((num_path,1)) for i, num_path in enumerate(data.num_paths)], dim=0)
        perturbed_x = x + z * std
        
        # Score
        score = model(data, random_t, perturbed_x)
        
        # Loss function
        loss = torch.mean(torch.sum((score * std + z)**2, dim=1)) # 4.x
        loss_tot += loss.item()

    loss_avg = loss_tot / len(loader)
    return loss_avg

def marginal_prob_std(t, sigma):
    r"""Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
    
    Returns:
        The standard deviation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t.to(device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))