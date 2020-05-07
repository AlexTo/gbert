from pytorch_pretrained_bert.modeling import BertPooler
from pytorch_transformers import BertPreTrainedModel

from src.models.BertEncoder import BertEncoder
from src.models.GBertEmbeddings import GBertEmbeddings


class GBert(BertPreTrainedModel):
    def __init__(self, config):
        super(GBert, self).__init__(config)
        self.embeddings = GBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def forward(self, embedding_inputs, head_mask=None, residual_h=None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_outputs = self.embeddings(embedding_inputs)
        encoder_outputs = self.encoder(embedding_outputs, head_mask=head_mask, residual_h=residual_h)
        sequence_output = encoder_outputs[0]
        pooler_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooler_output,) + encoder_outputs[1:]
        return outputs
