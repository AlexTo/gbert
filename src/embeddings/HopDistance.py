import networkx as nx
from tqdm import trange


class HopDistance(object):

    def __init__(self, nodes, links, batch_dict, k):
        self.k = k
        self.nodes = nodes
        self.links = links
        self.batch_dict = batch_dict

    def run(self):
        nodes = self.nodes
        links = self.links
        batch_dict = self.batch_dict

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(links)

        hop_dict = {}
        pbar = trange(0, len(batch_dict), desc="Computing hop distance")
        keys = list(batch_dict.keys())
        for i in pbar:
            node = keys[i]
            if node not in hop_dict:
                hop_dict[node] = {}
            for neighbor, score in batch_dict[node]:
                try:
                    hop = nx.shortest_path_length(G, source=node, target=neighbor)
                except:
                    hop = 99
                hop_dict[node][neighbor] = hop
        return hop_dict
