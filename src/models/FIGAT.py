import torch.nn.functional as f
from torch import nn

from src.models.GAT import GAT


class FIGAT(nn.Module):
    def __init__(self, feature_dim, type_ids, type_adj,
                 n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(FIGAT, self).__init__()
        self.gat = GAT(n_units, n_heads, dropout, attn_dropout, instance_normalization, diag)
        self.type_ids = type_ids
        self.type_adj = type_adj
        self.type_embedding = nn.Embedding(len(type_ids), n_units[-1])
        self.linear1 = nn.Linear(feature_dim, n_units[-1])

    def forward(self, x):
        ent_output = f.relu(self.linear1(x))
        type_emb = self.type_embedding(self.type_ids)
        attention_enhanced_type_emb = self.gat(type_emb, self.type_adj)
        output = ent_output.mm(attention_enhanced_type_emb.T)
        return output
