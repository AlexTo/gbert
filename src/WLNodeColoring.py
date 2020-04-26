import hashlib
from tqdm import trange


class WLNodeColoring(object):
    data = None
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    def __init__(self, nodes, links):
        self.nodes = nodes
        self.links = links

        for node in nodes:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        for pair in links:
            u1, u2 = pair
            if u1 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u1] = {}
            if u2 not in self.node_neighbor_dict:
                self.node_neighbor_dict[u2] = {}
            self.node_neighbor_dict[u1][u2] = 1
            self.node_neighbor_dict[u2][u1] = 1

    def run(self):
        nodes = self.nodes
        iteration_count = 1
        while True:
            new_color_dict = {}
            pbar = trange(0, len(nodes), desc=f"Computing WL iteration {iteration_count}")
            for i in pbar:
                node = nodes[i]
                neighbors = self.node_neighbor_dict[node]
                neighbor_color_list = [self.node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(self.node_color_dict[node])] + sorted(
                    [str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return self.node_color_dict
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1
