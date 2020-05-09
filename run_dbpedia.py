import argparse

from torch import optim
from torch.utils.data import DataLoader

from dbpedia_preprocess import load_data
from src.datasets.BatchWLHopPosDataset import *
from src.models.GBert import GBert
from src.models.GBertConfig import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DBP15K/zh_en", required=False,
                        help="Dataset directory: 'DBP15K/zh_en', 'DWY100K/dbp_wd', etc..")
    parser.add_argument("--lang_num", type=int, default=2, help="number of dataset languages, for e.g. 2 if fr and en")
    parser.add_argument("--k", type=int, default=7, help="number of k neighbors")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--train_split", type=float, default=0.3, help="training set split")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--node_embedding_dim", type=int, default=128, help="dimension of the node embedding layer")
    parser.add_argument("--rel_embedding_dim", type=int, default=128, help="dimension of the relation embedding layer")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimension of the hidden layer")
    parser.add_argument("--lr", type=float, default=0.005, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs to train")
    parser.add_argument("--check_point", type=int, default=100, help="check point")
    return parser.parse_args()


def initialize():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    return args, device


def configure(args, node_num):
    return GBertConfig(
        num_attention_heads=2,
        num_hidden_layers=2,
        embeddings={
            "neighbors": GBertEmbeddingConfig(num_embeddings=node_num),
            "wl": GBertEmbeddingConfig(num_embeddings=200),
            "hops": GBertEmbeddingConfig(num_embeddings=100),
            "pos_ids": GBertEmbeddingConfig(num_embeddings=args.k + 1),
        })


def main():
    args, device = initialize()

    input_dir = f"data/{args.dataset}"
    output_dir = f"outputs/{args.dataset}"
    num_lang = args.lang_num

    all_links, all_nodes, all_rels, pre_alignments, adj = load_data(input_dir, num_lang)

    np.random.shuffle(pre_alignments)
    train_pre_alignments = np.array(pre_alignments[:int(len(pre_alignments) // 1 * args.train_split)], dtype=np.int32)
    test_pre_alignments = np.array(pre_alignments[int(len(pre_alignments) // 1 * args.train_split):], dtype=np.int32)

    node_num = len(all_nodes)
    rel_num = len(np.unique(all_rels))

    config = configure(args, node_num)

    g_bert = GBert(config).to(device)

    dataset = GBertDataset(input_dir, output_dir, num_lang, args.k)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2)

    optimizer = optim.Adagrad(g_bert.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        g_bert.train()
        for embedding_inputs in loader:
            optimizer.zero_grad()
            output = g_bert(embedding_inputs)

            optimizer.step()


if __name__ == "__main__":
    main()
