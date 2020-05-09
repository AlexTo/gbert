import os

import numpy as np
import scipy.sparse as sp
import torch


def torch_adj_normalize(mx):
    row_sum = torch.sum(mx, dim=1)
    r_inv = row_sum.pow(-0.5)
    r_inv[torch.isinf(r_inv)] = 0
    r_mat_inv = torch.diag(r_inv)
    return r_mat_inv.mm(mx).mm(r_mat_inv)


def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    row_sum = np.array(mx.sum(1))
    r_inv = np.power(row_sum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def open_file(path, file, mode):
    if not os.path.exists(path):
        os.makedirs(path)
    return open(f"{path}/{file}", mode)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def batched_target_to_adj(targets):
    batch_size = targets.size()[0]
    n = targets.size()[1]
    s_target = torch.reshape(targets, [batch_size * n])
    zero = torch.zeros(batch_size * n)
    z_target = torch.where(s_target < 0, s_target, zero)
    indices = torch.nonzero(z_target, as_tuple=True)[0]
    s_mask_adj = torch.ones((batch_size * n, n))
    s_mask_adj[indices] = 0
    mask_adj = torch.reshape(s_mask_adj, [batch_size, n, n])
    mask_adj_r = torch.rot90(mask_adj, 1, [1, 2])
    adj = (mask_adj.bool() & mask_adj_r.bool() & torch.logical_not(
        torch.diag_embed(torch.ones((batch_size, n))).bool())).float()
    return adj


def batched_adj_to_freq(adj):
    return torch.sum(adj, dim=0)
