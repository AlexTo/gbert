import argparse
import pickle

from scipy.linalg import inv

from src.HopDistance import *
from src.GraphBatching import *
from src.WLNodeColoring import *
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
