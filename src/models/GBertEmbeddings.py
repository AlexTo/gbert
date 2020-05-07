import torch
from torch import nn


class GBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(GBertEmbeddings, self).__init__()
        self.embeddings = nn.ModuleDict({
            k: nn.Embedding(num_embeddings=v.num_embeddings,
                            embedding_dim=config.hidden_size) for k, v in config.embeddings.items()
        })
        self.fusion = config.fusion
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, embedding_inputs):
        embedding_outputs = [self.embeddings[k](v.to(self.embeddings[k].weight.device)) for k, v in
                             embedding_inputs.items()]
        output = self.fusion(embedding_outputs)
        output = self.layer_norm(output)
        output = self.dropout(output)
        return output
