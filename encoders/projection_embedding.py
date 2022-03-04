import torch
import torch.nn as nn
import statistics
import torchvision.models as models


class ImgTextEmbeddings(nn.Module):
    def __int__(self, projection_dims, dropout_rate, proj_layers):
        super(ImgTextEmbeddings, self).__init__()
        self.Linear_unit = nn.Linear(projection_dims)
        self.Gelu = nn.GELU()
        self.Dense_unit = nn.Linear(projection_dims)
        self.dropout = nn.Dropout(dropout_rate)
        self.addition = nn.add()
        self.layer_norm = nn.LayerNorm()
        self.projection_layers = proj_layers

    def forward(self, embeddings):
        embed_proj = self.Linear_unit(embeddings)
        for _ in range(self.projection_layers):
            x = self.Gelu(embed_proj)
            x = self.Dense_unit(x)
            x = self.dropout(x)
            x = self.addition(embed_proj, x)
            embed_proj = self.layer_norm(x)
        return embed_proj
        




# def Projection_emb_Keras(embeddings, num_projection_layers, projection_dims, dropout_rate):
    
#     projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
#     for _ in range(num_projection_layers):
#         x = tf.nn.gelu(projected_embeddings)
#         x = layers.Dense(projection_dims)(x)
#         x = layers.Dropout(dropout_rate)(x)
#         x = layers.Add()([projected_embeddings, x])
#         projected_embeddings = layers.LayerNormalization()(x)
#     return projected_embeddings