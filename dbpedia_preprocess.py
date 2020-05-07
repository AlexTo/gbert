import argparse
import pickle

from scipy.linalg import inv

from src.embeddings.GraphBatching import *
from src.embeddings.HopDistance import *
from src.embeddings.WLNodeColoring import *
from src.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DBP15K/zh_en", required=False,
                        help="Dataset directory: 'DBP15K/zh_en', 'DWY100K/dbp_wd', etc..")
    parser.add_argument("--c", type=float, default=0.15, help="c factor for intimacy matrix S calculation")
    parser.add_argument("--cuda", action="store_true", default=True, help="Whether to use cuda")
    parser.add_argument("--lang_num", type=int, default=2, help="number of dataset languages, for e.g. 2 if fr and en")
    parser.add_argument("--batching", action="store_true", default=False, help="Whether to run batching pre-processing")
    parser.add_argument("--wl", action="store_true", default=False,
                        help="Whether to run Weisfeiler Lehman pre-processing")
    parser.add_argument("--hop_distance", action="store_true", default=False,
                        help="Whether to run Hop distance pre-processing")
    return parser.parse_args()


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
    rel_sets = []
    for path in paths:
        triples = np.genfromtxt(path, dtype=np.int32)
        link_set = triples[:, [0, 2]]
        pred_set = triples[:, 1]
        link_sets.append(link_set)
        rel_sets.append(pred_set)
    return link_sets, rel_sets


def load_pre_alignments(input_dir):
    path = input_dir + "/ill_ent_ids"
    pre_alignments = np.genfromtxt(path, dtype=np.int32)
    return pre_alignments


def load_data(input_dir, lang_num):
    node_sets = load_nodes(input_dir, lang_num)
    link_sets, rel_sets = load_links(input_dir, lang_num)
    pre_alignments = load_pre_alignments(input_dir)

    all_links = np.concatenate(link_sets)
    all_nodes = np.concatenate(node_sets)

    all_rels = np.concatenate(rel_sets)

    num_links = all_links.shape[0]
    num_nodes = all_nodes.shape[0]
    adj = sp.coo_matrix((np.ones(num_links * 2),
                         (np.concatenate([all_links[:, 0], all_links[:, 1]]),
                          np.concatenate([all_links[:, 1], all_links[:, 0]]))),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    return all_links, all_nodes, all_rels, pre_alignments, adj


def main():
    args = parse_args()
    input_dir = f"data/{args.dataset}"
    output_dir = f"outputs/{args.dataset}"
    num_lang = args.lang_num
    c = args.c

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    all_links, all_nodes, all_rels, pre_alignments, adj = load_data(input_dir, num_lang)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    norm_adj = adj_normalize(adj + sp.eye(adj.shape[0]))

    if args.batching:
        # should we include pre-alignments in this process?
        if os.path.exists(f"{output_dir}/eigen_adj.bin"):
            with open(f"{output_dir}/eigen_adj.bin", 'rb') as f:
                eigen_adj = pickle.load(f)
        else:
            eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * adj_normalize(adj)).toarray())
            with open_file(output_dir, "eigen_adj.bin", 'wb') as f:
                pickle.dump(eigen_adj, f, protocol=pickle.HIGHEST_PROTOCOL)

        for k in [7]:
            if os.path.exists(f"{output_dir}/batching/batch_dict_{k}.bin"):
                print(f"batch_dict_{k}.bin exists. Skipping")
                continue
            graph_batching = GraphBatching(all_nodes, eigen_adj, k)
            batch_dict = graph_batching.run()
            with open_file(f"{output_dir}/batching", f"batch_dict_{k}.bin", "wb") as f:
                pickle.dump(batch_dict, f)

    if args.wl:
        # should we include pre-alignments in this process?
        if not os.path.exists(f"{output_dir}/wl/wl.bin"):
            wl_node_coloring = WLNodeColoring(all_nodes, all_links)
            wl = wl_node_coloring.run()
            with open_file(f"{output_dir}/wl", "wl.bin", "wb") as f:
                pickle.dump(wl, f)
        else:
            print("WL file exists. Skipping")

    if args.hop_distance:
        for k in [7]:
            if os.path.exists(f"{output_dir}/hop/hop_dict_{k}.bin"):
                print(f"hop_dict_{k}.bin exists. Skipping")
                continue
            with open(f"{output_dir}/batching/batch_dict_{k}.bin", 'rb') as f:
                batches = pickle.load(f)
            hop_distance = HopDistance(all_nodes, all_links, batches, k)
            hop_dict = hop_distance.run()
            with open_file(f"{output_dir}/hop", f"hop_dict_{k}.bin", "wb") as f:
                pickle.dump(hop_dict, f)


if __name__ == "__main__":
    main()
