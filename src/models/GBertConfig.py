from pytorch_transformers import PretrainedConfig

from src.fusions.SumFusion import SumFusion


class GBertConfig(PretrainedConfig):
    def __init__(self,
                 embeddings={},
                 fusion=SumFusion(),
                 hidden_size=128,
                 num_hidden_layers=1,
                 num_attention_heads=1,
                 intermediate_size=128,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.5,
                 attention_probs_dropout_prob=0.3,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 is_decoder=False,
                 **kwargs):
        super(GBertConfig, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.fusion = fusion
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder


class GBertEmbeddingConfig(object):
    def __init__(self, num_embeddings):
        self.num_embeddings = num_embeddings
