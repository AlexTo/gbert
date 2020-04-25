import argparse

import torch
from scipy.linalg import inv
from src.utils import *
from src.WLNodeColoring import *
from src.GraphBatching import *
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en", required=False,
                        help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--c", type=float, default=0.15, help="c factor for intimacy matrix S calculation")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--lang_num", type=int, default=2, help="number of dataset languages, for e.g. 2 if fr and en")
    parser.add_argument("--batching", action="store_true", default=True,
                        help="Whether to run batching pre-processing")
    parser.add_argument("--wl", action="store_true", default=True,
                        help="Whether to run Weisfeiler Lehman pre-processing")
    return parser.parse_args()


def load_nodes(file_dir, lang_num):
    paths = [file_dir + "/ent_ids_" + str(i) for i in range(1, lang_num + 1)]
    node_sets = []
    for path in paths:
        node_set = np.genfromtxt(path, dtype=np.int32)[:, 0]
        node_sets.append(node_set)
    return node_sets


def load_links(file_dir, lang_num):
    paths = [file_dir + "/triples_" + str(i) for i in range(1, lang_num + 1)]
    link_sets = []
    pred_sets = []
    for path in paths:
        triples = np.genfromtxt(path, dtype=np.int32)
        link_set = triples[:, [0, 2]]
        pred_set = triples[:, 1]
        link_sets.append(link_set)
        pred_sets.append(pred_set)
    return link_sets, pred_sets


def load_data(file_dir, lang_num):
    node_sets = load_nodes(file_dir, lang_num)
    link_sets, pred_sets = load_links(file_dir, lang_num)
    all_links = np.concatenate(link_sets)
    all_nodes = np.concatenate(node_sets)

    num_links = all_links.shape[0]
    num_nodes = all_nodes.shape[0]
    adj = sp.coo_matrix((np.ones(num_links * 2),
                         (np.concatenate([all_links[:, 0], all_links[:, 1]]),
                          np.concatenate([all_links[:, 1], all_links[:, 0]]))),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    return all_links, all_nodes, adj


def main():
    args = parse_args()
    file_dir = args.file_dir
    num_lang = args.lang_num
    c = args.c
    run_wl = args.wl
    run_batching = args.batching

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    all_links, all_nodes, adj = load_data(file_dir, num_lang)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    norm_adj = adj_normalize(adj + sp.eye(adj.shape[0]))

    if run_batching:
        # should we include pre-alignments in this process?
        eigen_adj_before_inv = sparse_mx_to_torch_sparse_tensor(sp.eye(adj.shape[0]) - (1 - c) * norm_adj)
        eigen_adj_before_inv.to(device)

        eigen_adj = c * torch.inverse(eigen_adj_before_inv)

        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            graph_batching = GraphBatching(all_nodes, eigen_adj, k)
            batches = graph_batching.run()
            f = open_file(f"{file_dir.replace('data', 'outputs')}/batching", f"batch_{k}.bin", "wb")
            pickle.dump(batches, f)

    if 0:
        # should we include pre-alignments in this process?
        wl_node_coloring = WLNodeColoring(all_nodes, all_links)
        wl = wl_node_coloring.run()
        f = open_file(f"{file_dir.replace('data', 'outputs')}/wl", "wl.bin", "wb")
        pickle.dump(wl, f)


if __name__ == "__main__":
    main()
