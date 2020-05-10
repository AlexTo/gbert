import argparse
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn

from torch.utils.data import DataLoader
from tqdm import trange

from src.datasets.FigerDataset import FigerDataset
from src.models.FIGAT import FIGAT

import neptune

from src.utils import open_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/FIGER", required=False)
    parser.add_argument("--batch_size", type=int, default=102, help="Batch size")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--feature_dim", type=int, default=200)
    parser.add_argument("--hidden_units", type=str, default="200,200,200",
                        help="hidden units in each hidden layer(including in_dim and out_dim), split with comma")
    parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, split with comma")
    parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay (L2 loss on parameters)")
    parser.add_argument("--instance_normalization", action="store_true", default=False,
                        help="enable instance normalization")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
    parser.add_argument("--seed", type=int, default=23455, help="random seed")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    return parser.parse_args()


def initialize():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    return args, device


def evaluate(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    batches = 0
    with torch.no_grad():
        for _, features, targets in loader:
            features, targets = features.float().to(device), targets.float().to(device)
            batches += 1
            outputs = model(features)
            batch_loss = criterion(outputs, targets)
            loss += batch_loss.item()
    loss = loss / batches
    return loss


def save_result(output_dir, model, loader, device, ds="dev"):
    model.eval()
    results = []

    with torch.no_grad():
        for entities, features, targets in loader:
            features, targets = features.float().to(device), targets.float().to(device)
            outputs = model(features)
            results.append(np.concatenate((np.array(entities).reshape(len(entities), -1), outputs.cpu()), axis=1))

    with open_file(f"{output_dir}", f"{ds}.probs", "w") as f:
        results = np.vstack(results)
        np.savetxt(f, results, fmt="%s", delimiter=' ')


def train():
    args, device = initialize()
    neptune.init('alexto/EMNLP2020')
    params = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs
    }
    neptune.create_experiment(name='Switch to Adagrad, update learning rate', params=params)

    n_units = [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    with open(f"{args.output_dir}/type_adj.bin", 'rb') as f:
        type_adj = pickle.load(f)

    num_types = type_adj.shape[0]
    type_adj = torch.tensor(type_adj).to_sparse()
    train_set = FigerDataset(args.output_dir, "train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=2)

    dev_set = FigerDataset(args.output_dir, "dev")
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=2)

    type_ids = torch.tensor(np.arange(num_types)).to(device)

    model = FIGAT(feature_dim=args.feature_dim, type_ids=type_ids, type_adj=type_adj,
                  n_units=n_units, n_heads=n_heads, dropout=args.dropout, attn_dropout=args.attn_dropout,
                  instance_normalization=args.instance_normalization, diag=True).to(device)

    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    pbar = trange(1, args.epochs + 1, desc="Training")
    dev_loss = 9999
    for _ in pbar:
        model.train()
        for _, features, targets in train_loader:
            features, targets = features.float().to(device), targets.float().to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"train_loss": f"{loss:.4f}", "dev_loss": f"{dev_loss:.4f}"})
            neptune.log_metric("train_loss", loss)
        dev_loss = evaluate(model, criterion, dev_loader, device)
        neptune.log_metric("dev_loss", dev_loss)

    torch.save(model.state_dict(), "figer_model.pt")


def test():
    args, device = initialize()

    n_units = [int(x) for x in args.hidden_units.strip().split(",")]
    n_heads = [int(x) for x in args.heads.strip().split(",")]

    with open(f"{args.output_dir}/type_adj.bin", 'rb') as f:
        type_adj = pickle.load(f)

    num_types = type_adj.shape[0]
    type_ids = torch.tensor(np.arange(num_types)).to(device)
    type_adj = torch.tensor(type_adj).to_sparse()

    dev_set = FigerDataset(args.output_dir, "dev")
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=2)

    test_set = FigerDataset(args.output_dir, "test")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=2)

    model = FIGAT(feature_dim=args.feature_dim, type_ids=type_ids, type_adj=type_adj,
                  n_units=n_units, n_heads=n_heads, dropout=args.dropout, attn_dropout=args.attn_dropout,
                  instance_normalization=args.instance_normalization, diag=True).to(device)

    model.load_state_dict(torch.load("figer_model.pt"))
    save_result(args.output_dir, model, dev_loader, device, ds="dev")
    save_result(args.output_dir, model, test_loader, device, ds="test")


if __name__ == "__main__":
    # train()
    test()
