from torch.utils.data import Dataset
from src.utils import *
import pickle

class GBertDataset(Dataset):
  def __init__(self, dataset, num_lang, k):
    self.input_dir = f"data/{dataset}"
    self.output_dir = f"outputs/{dataset}"
    self.k = k
    self.num_lang = num_lang
    self.all_nodes = load_nodes(self.input_dir, num_lang)
    self.wl = pickle.load(open(f"{self.output_dir}/wl/wl.bin", 'rb'))
    self.batch_dict = pickle.load(open(f"{self.output_dir}/batching/batch_dict_{k}.bin", 'rb'))
    self.hop_dict = pickle.load(open(f"{self.output_dir}/hop/hop_dict_{k}.bin", 'rb'))
    
  def __len__(self):
    return len(self.all_nodes)

  def __getitem__(self, idx):
    node = self.all_nodes[idx]
    neighbors = self.batch_dict[node]
    
    hops = self.hop_dict[node]
    hops.insert(0, 0)
    
    wl = [self.wl[node]]    
    for n in neighbors:
      wl.append(self.wl[n])
    
    neighbors.insert(0, node)
    
    pos_ids = range(len(neighbors_list) + 1)
    
    return (neighbors, wl, hops, pos_ids)