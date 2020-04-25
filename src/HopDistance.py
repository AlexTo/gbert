import networkx as nx
import pickle

class MethodHopDistance(object):
  
    def __init__(self, k, nodes, links, batch_dict):
      self.k = k
      self.nodes = nodes
      self.links = links

    def run(self):
      nodes = self.nodes
      links = self.links
      G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(links)

        hop_dict = {}
        for node in batch_dict:
            if node not in hop_dict: hop_dict[node] = {}
            for neighbor, score in batch_dict[node]:
                try:
                    hop = nx.shortest_path_length(G, source=node, target=neighbor)
                except:
                    hop = 99
                hop_dict[node][neighbor] = hop
        return hop_dict