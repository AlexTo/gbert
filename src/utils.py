import numpy as np
import scipy.sparse as sp
import os

import torch


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
