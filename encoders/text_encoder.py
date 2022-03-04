import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                    hidden_size=256,
                                    num_hidden_layers=1,
                                    num_attention_heads=1,
                                    intermediate_size=512,
                                    hidden_act="gelu",
                                    hidden_dropout_prob=0.1,
                                    attention_probs_dropout_prob=0.1,
                                    max_position_embeddings=512,
                                    type_vocab_size=2,
                                    initializer_range=0.02,
                                    layer_norm_eps=1e-12,
                                    is_decoder=False)

        encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, 256, padding_idx=src_tokenizer.pad_token_id)
        self.encoder = BertModel(encoder_config)
        self.encoder.set_input_embeddings(encoder_embeddings)

    def forward(self, encoder_input_ids):
        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        return encoder_hidden_states


class LSTMEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, dropout_prob):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights(self.fc)

    def init_weights(self, m):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)

    def forward(self, input_ids):
        embedded_captions = self.embedding(input_ids)
        hidden_outputs, _ = self.lstm(embedded_captions)
        outputs = self.fc(hidden_outputs)
        return outputs