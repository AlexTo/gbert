import argparse
from random import random
from torch import optim
from models import GBertConfig, GBert
from src.utils import *
from src.GBert import *
from src.GBertDataset import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="DBP15K/zh_en", required=False,
                        help="Dataset directory: 'DBP15K/zh_en', 'DWY100K/dbp_wd', etc..")
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

    all_links, all_nodes, adj = load_data(input_dir, num_lang)
    
    np.random.shuffle(pre_alignments)
    train_pre_alignments = np.array(pre_alignments[:int(len(pre_alignments) // 1 * args.train_split)], dtype=np.int32)
    test_pre_alignments = np.array(pre_alignments[int(len(pre_alignments) // 1 * args.train_split):], dtype=np.int32)

    ent_num = len(all_nodes)
    rel_num = len(pred_subjs)
    
    config = GBertConfig(ent_num, rel_num, args.entity_embedding_dim, args.entity_embedding_dim)

    g_bert = GBert(config).to(device)

    optimizer = optim.Adagrad(g_bert.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        
        g_bert.train()
        optimizer.zero_grad()
        output = g_bert(entity_indices)

        # Loss function, cost function goes here

        optimizer.step()

        # Test model and print some results here


if __name__ == "__main__":
    main()
