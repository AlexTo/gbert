from pytorch_transformers import PretrainedConfig
from torch import nn
from transformers.modeling_bert import BertPreTrainedModel


class GBertConfig(PretrainedConfig):
    def __init__(self, node_num,
                 rel_num,
                 feature_embedding_dim=128,
                 rel_embedding_dim=128,
                 hidden_dim=128,
                 layer_norm_eps=1e-12,
                 max_wl_index=100,
                 max_hop_dis_index=100,
                 max_pos_index=100,
                 is_decoder=False, **kwargs):
        super(GBertConfig, self).__init__(**kwargs)
        self.node_num = node_num
        self.rel_num = rel_num
        self.feature_embedding_dim = feature_embedding_dim
        self.rel_embedding_dim = rel_embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_norm_eps = layer_norm_eps


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.rel_embedding = nn.Embedding(num_embeddings=config.rel_num,
                                          embedding_dim=config.rel_embedding_dim)
        self.neighbors_embedding = nn.Embedding(num_embeddings=config.node_num,
                                                embedding_dim=config.feature_embedding_dim)
        self.wl_embedding = nn.Embedding(num_embeddings=config.max_wl_index,
                                         embedding_dim=config.feature_embedding_dim)
        self.hop_embedding = nn.Embedding(num_embeddings=config.max_hop_dis_index,
                                          embedding_dim=config.feature_embedding_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=config.max_pos_index,
                                          embedding_dim=config.feature_embedding_dim)

        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, neighbors, wl, hops, pos_ids):
        neighbors_embedding = self.neighbors_embedding(neighbors)
        wl_embedding = self.wl_embedding(wl)
        hop_embedding = self.hop_embedding(hops)
        pos_embedding = self.pos_embedding(pos_ids)
        embeddings = neighbors_embedding + wl_embedding + hop_embedding + pos_embedding
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()


class GBert(BertPreTrainedModel):
    def __init__(self, config):
        super(GBert, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self, neighbors, wl, hops, pos_ids):
        embeddings = self.embeddings(neighbors, wl, hops, pos_ids)
