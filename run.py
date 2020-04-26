import argparse
from random import random
from torch import optim
from models import GBertConfig, GBert
from torch.utils.data import DataLoader

from src.utils import *
from src.GBert import *
from src.GBertDataset import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DBP15K/zh_en", required=False,
                        help="Dataset directory: 'DBP15K/zh_en', 'DWY100K/dbp_wd', etc..")
    parser.add_argument("--lang_num", type=int, default=2, help="number of dataset languages, for e.g. 2 if fr and en")
    parser.add_argument("--k", type=int, default=7, help="number of k neighbors")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--train_split", type=float, default=0.3, help="training set split")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--node_embedding_dim", type=int, default=128, help="dimension of the node embedding layer")
    parser.add_argument("--rel_embedding_dim", type=int, default=128, help="dimension of the relation embedding layer")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimension of the hidden layer")
    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check_point", type=int, default=100, help="check point")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_dir = f"data/{args.dataset}"
    output_dir = f"outputs/{args.dataset}"
    num_lang = args.lang_num

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    all_links, all_nodes, all_rels, pre_alignments, adj = load_data(input_dir, num_lang)

    np.random.shuffle(pre_alignments)
    train_pre_alignments = np.array(pre_alignments[:int(len(pre_alignments) // 1 * args.train_split)], dtype=np.int32)
    test_pre_alignments = np.array(pre_alignments[int(len(pre_alignments) // 1 * args.train_split):], dtype=np.int32)

    node_num = len(all_nodes)
    rel_num = len(np.unique(all_rels))

    config = GBertConfig(node_num, rel_num, args.node_embedding_dim, args.rel_embedding_dim)

    g_bert = GBert(config).to(device)

    dataset = GBertDataset(input_dir, output_dir, num_lang, args.k)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2)

    optimizer = optim.Adagrad(g_bert.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        g_bert.train()
        for neighbors, wl, hops, pos_ids in loader:
            neighbors, wl, hops, pos_ids = neighbors.to(device), wl.to(device), hops.to(device), pos_ids.to(device)
            optimizer.zero_grad()
            output = g_bert(neighbors, wl, hops, pos_ids)
            # Loss function, cost function goes here
            optimizer.step()


if __name__ == "__main__":
    main()
