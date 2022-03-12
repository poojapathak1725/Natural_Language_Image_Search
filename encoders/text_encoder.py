import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

#         self.src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = vocab_size

        encoder_config = BertConfig(vocab_size=self.vocab_size,
                                    hidden_size=512,
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

        encoder_embeddings = torch.nn.Embedding(self.vocab_size, 512, padding_idx=0)
        self.encoder = BertModel(encoder_config)
        self.encoder.set_input_embeddings(encoder_embeddings)
        
        self.Linear_unit = nn.Linear(512, 300)
        self.Gelu = nn.GELU()
        self.Dense_unit = nn.Linear(300, 300)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(300)

    def forward(self, encoder_input_ids):
        encoder_hidden_states = self.encoder(encoder_input_ids)[1]       
        embed_proj = self.Linear_unit(encoder_hidden_states)
        
        x = self.Gelu(embed_proj)
        x = self.Dense_unit(x)
#         x = self.dropout(x)
        x = torch.add(embed_proj, x)
        embed_proj = self.layer_norm(x)
        
        
        return embed_proj



class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed_size = 300
        self.hidden_size = 512
        self.vocab_size = vocab_size
        self.num_layers = 1
        self.dropout_prob = 0.1

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True, dropout=self.dropout_prob)
        self.fc = nn.Linear(self.hidden_size, 300)
        
        self.Linear_unit = nn.Linear(self.vocab_size, 300)
        self.Relu = nn.ReLU()
        self.Dense_unit = nn.Linear(300, 300)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(300)
        self.init_weights(self.fc)

    def init_weights(self, m):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)

    def forward(self, input_ids):
        
        
        embedded_captions = self.embedding(input_ids)
        hidden_outputs, _ = self.lstm(embedded_captions)
        hidden_outputs = hidden_outputs[:, -1, :]
        outputs = self.fc(hidden_outputs)
        
#         embed_proj = self.Linear_unit(outputs)
        
#         x = self.Relu(embed_proj)
# #         x = self.Dense_unit(x)
#         x = self.dropout(x)
#         x = torch.add(embed_proj, x)
#         embed_proj = self.layer_norm(x)
        
        return outputs