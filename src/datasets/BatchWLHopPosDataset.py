from torch.utils.data import Dataset
from src.utils import *
import pickle


class GBertDataset(Dataset):
    def __init__(self, input_dir, output_dir, num_lang, k):
        self.all_nodes = np.concatenate(load_nodes(input_dir, num_lang))
        self.wl = pickle.load(open(f"{output_dir}/wl/wl.bin", 'rb'))
        self.batch_dict = pickle.load(open(f"{output_dir}/batching/batch_dict_{k}.bin", 'rb'))
        self.hop_dict = pickle.load(open(f"{output_dir}/hop/hop_dict_{k}.bin", 'rb'))

    def __len__(self):
        return len(self.all_nodes)

    def __getitem__(self, idx):
        node = self.all_nodes[idx]
        neighbors = [n[0] for n in self.batch_dict[node]]

        hops = [v for k, v in self.hop_dict[node].items()]
        hops.insert(0, 0)

        neighbors.insert(0, node)

        wl = [self.wl[n] for n in neighbors]

        pos_ids = range(len(neighbors))

        neighbors = torch.LongTensor(neighbors)
        pos_ids = torch.LongTensor(pos_ids)
        wl = torch.LongTensor(wl)
        hops = torch.LongTensor(hops)

        return {"neighbors": neighbors, "wl": wl, "hops": hops, "pos_ids": pos_ids}
