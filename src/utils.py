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


def load_nodes(input_dir, lang_num):
    paths = [input_dir + "/ent_ids_" + str(i) for i in range(1, lang_num + 1)]
    node_sets = []
    for path in paths:
        node_set = np.genfromtxt(path, dtype=np.int32)[:, 0]
        node_sets.append(node_set)
    return node_sets


def load_links(input_dir, lang_num):
    paths = [input_dir + "/triples_" + str(i) for i in range(1, lang_num + 1)]
    link_sets = []
    pred_sets = []
    for path in paths:
        triples = np.genfromtxt(path, dtype=np.int32)
        link_set = triples[:, [0, 2]]
        pred_set = triples[:, 1]
        link_sets.append(link_set)
        pred_sets.append(pred_set)
    return link_sets, pred_sets

  
def load_pre_alignments(input_dir):
    path = input_dir + "/ill_ent_ids"
    
    
    
def load_data(input_dir, lang_num):
    node_sets = load_nodes(input_dir, lang_num)
    link_sets, pred_sets = load_links(input_dir, lang_num)
    all_links = np.concatenate(link_sets)
    all_nodes = np.concatenate(node_sets)

    num_links = all_links.shape[0]
    num_nodes = all_nodes.shape[0]
    adj = sp.coo_matrix((np.ones(num_links * 2),
                         (np.concatenate([all_links[:, 0], all_links[:, 1]]),
                          np.concatenate([all_links[:, 1], all_links[:, 0]]))),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    return all_links, all_nodes, adj
