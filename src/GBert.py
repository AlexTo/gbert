from pytorch_transformers import PretrainedConfig
from torch import nn
from transformers.modeling_bert import BertPreTrainedModel


class GBertConfig(PretrainedConfig):
    def __init__(self, ent_num, rel_num, entity_embedding_dim, rel_embedding_dim, hidden_dim, layer_norm_eps=1e-12,
                 is_decoder=False, **kwargs):
        super(GBertConfig, self).__init__(**kwargs)


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.entity_embedding = nn.Embedding(num_embeddings=config.ent_num,
                                             embedding_dim=config.entity_embedding_dim)
        self.relation_embedding = nn.Embedding(num_embeddings=config.rel_num,
                                               embedding_dim=config.rel_embedding_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()


class GBert(BertPreTrainedModel):
    def __init__(self, config):
        super(GBert, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self, ):
        
