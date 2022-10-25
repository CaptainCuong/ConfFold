import torch
from torch_scatter import scatter_add
import copy

def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1).reshape(-1,1)


def eq_transform(score_d, pos, edge_index, edge_length):
    N = pos.size(0)
    dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (E, 3)
    score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
        + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    return score_pos


def convert_cluster_score_d(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length, subgraph_index):
    """
    Args:
        cluster_score_d:    (E_c, 1)
        subgraph_index:     (N, )
    """
    cluster_score_pos = eq_transform(cluster_score_d, cluster_pos, cluster_edge_index, cluster_edge_length)  # (C, 3)
    score_pos = cluster_score_pos[subgraph_index]
    return score_pos


def get_cos_angle_from_ind(pos, vec1, vec2):
    """
    Args:
        pos:  (N, 3)
        vec1:  (2, A), u1 --> v1, 
        vec2:  (2, A), u2 --> v2,
    """
    u1, v1 = vec1
    u2, v2 = vec2
    vec1 = pos[v1] - pos[u1] # (A, 3)
    vec2 = pos[v2] - pos[u2]
    inner_prod = torch.sum(vec1 * vec2, dim=-1, keepdim=True)   # (A, 1)
    length_prod = torch.norm(vec1, dim=-1, keepdim=True) * torch.norm(vec2, dim=-1, keepdim=True)   # (A, 1)
    cos_angle = inner_prod / length_prod    # (A, 1)
    
    assert torch.isnan(cos_angle).sum().item() == 0
    return cos_angle

def get_cos_angle_from_vec(vec1, vec2):
    """
    Args:
        vec1:  (A, 3)
        vec2:  (A, 3)
    Return:
        Shape: (A, 1)
    """
    inner_prod = torch.sum(vec1 * vec2, dim=-1, keepdim=True)   # (A, 1)
    length_prod = torch.norm(vec1, dim=-1, keepdim=True) * torch.norm(vec2, dim=-1, keepdim=True)   # (A, 1)
    cos_angle = inner_prod / length_prod    # (A, 1)
    
    assert torch.isnan(cos_angle).sum().item() == 0
    return cos_angle

def get_pseudo_vec(pos, vec1, vec2):
    """
    Args:
        pos:  (N, 3)
        vec1:  (2, E), u1 --> v1, 
        vec2:  (2, E), u2 --> v2, 
    """
    u1, v1 = vec1
    u2, v2 = vec2
    vec1 = pos[v1] - pos[u1] # (E, 3)
    vec2 = pos[v2] - pos[u2] # (E, 3)

    return torch.cross(vec1, vec2, dim=1)

def get_dihedral(pos, dihedral_index):
    """
    Args:
        pos:  (N, 3)
        dihedral:  (4, A)
    """
    n1, ctr1, ctr2, n2 = dihedral_index # (A, )
    v_ctr = pos[ctr2] - pos[ctr1]   # (A, 3)
    v1 = pos[n1] - pos[ctr1]
    v2 = pos[n2] - pos[ctr2]
    n1 = torch.cross(v_ctr, v1, dim=-1) # Normal vectors of the two planes
    n2 = torch.cross(v_ctr, v2, dim=-1)
    inner_prod = torch.sum(n1 * n2, dim=1, keepdim=True)    # (A, 1)
    length_prod = torch.norm(n1, dim=-1, keepdim=True) * torch.norm(n2, dim=-1, keepdim=True)
    dihedral = torch.acos(inner_prod / length_prod)    # (A, 1)
    
    assert torch.isnan(dihedral).sum().item() == 0
    return dihedral

def get_direction(pos, direction_frame):
    """
    Args:
        pos:  (N, 3)
        direction_frame:  (D, 4)
            Row1: Node need to calculate direction
            Row2: Node O
            Row3: Node x
            Row4: Node y
    Return:
        Shape: (D, 3)
    """
    n_ind, o_ind, x_ind, y_ind = direction_frame.T
    vec_On = pos[n_ind] - pos[o_ind]
    vec_Ox = pos[x_ind] - pos[o_ind]
    vec_Oy = pos[y_ind] - pos[o_ind]
    vec_Oz = torch.cross(vec_Ox, vec_Oy, dim=1)
    vec_Oy = torch.cross(vec_Oz, vec_Ox, dim=1)
    angle_x = get_cos_angle_from_vec(vec_On, vec_Ox)
    angle_y = get_cos_angle_from_vec(vec_On, vec_Oy)
    angle_z = get_cos_angle_from_vec(vec_On, vec_Oz)

    return torch.cat([angle_x, angle_y, angle_z], dim=1)