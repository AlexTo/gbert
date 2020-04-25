import argparse
from random import random
from torch import optim
from models import GBertConfig, GBert
from src.utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en", required=False,
                        help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
    parser.add_argument("--lang_num", type=int, default=2, help="number of dataset languages, for e.g. 2 if fr and en")
    parser.add_argument("--train_split", type=float, default=0.3, help="training set split")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--entity_embedding_dim", type=int, default=128, help="dimension of the entity embedding layer")
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

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    ent2id_dict, pre_alignments, triples, pred_subjs, pred_objs, id_lists = read_data(args.file_dir, args.lang_num)
    np.random.shuffle(pre_alignments)
    train_pre_alignments = np.array(pre_alignments[:int(len(pre_alignments) // 1 * args.train_split)], dtype=np.int32)
    test_pre_alignments = np.array(pre_alignments[int(len(pre_alignments) // 1 * args.train_split):], dtype=np.int32)

    ent_num = len(ent2id_dict)
    rel_num = len(pred_subjs)
    config = GBertConfig(ent_num, rel_num, args.entity_embedding_dim, args.entity_embedding_dim)

    adjacency_matrix = get_adjacency_matrix(ent_num, triples, norm=True).to(device)

    g_bert = GBert(config).to(device)

    optimizer = optim.Adagrad(g_bert.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    entity_indices = torch.LongTensor(np.arange(ent_num)).to(device)

    for epoch in range(args.epochs):
        g_bert.train()
        optimizer.zero_grad()
        output = g_bert(entity_indices)

        # Loss function, cost function goes here

        optimizer.step()

        # Test model and print some results here


if __name__ == "__main__":
    main()
