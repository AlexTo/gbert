class GraphBatching(object):
    def __init__(self, nodes, eigen_adj, k):
        self.k = k
        self.eigen_adj = eigen_adj
        self.nodes = nodes

    def run(self):
        eigen_adj = self.eigen_adj
        nodes = self.nodes

        user_top_k_neighbor_intimacy_dict = {}
        for node_id in nodes:
            s = eigen_adj[node_id]
            s[node_id] = -1000.0
            top_k_neighbor_index = s.argsort()[-self.k:][::-1]
            user_top_k_neighbor_intimacy_dict[node_id] = []
            for neighbor_index in top_k_neighbor_index:
                neighbor_id = nodes[neighbor_index]
                user_top_k_neighbor_intimacy_dict[node_id].append((neighbor_id, s[neighbor_index]))
        return user_top_k_neighbor_intimacy_dict
