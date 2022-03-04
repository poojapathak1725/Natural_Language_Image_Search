from encoders.image_encoder import CNNEncoder
from encoders.text_encoder import BERTEncoder
from encoders.projection_embedding import ImgTextEmbeddings
import torch.nn as nn
import torch

class DualEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.text_encoder = BERTEncoder()
        self.image_encoder = CNNEncoder()
        self.projectionEmbeddings = ImgTextEmbeddings(
            configs["projection_dim"],
            configs["dropout_rate"],
            configs['projection_layers']
        )

    def forward(self, images, captions):

        encoded_images = self.projectionEmbeddings(
            self.image_encoder(images)
        )

        encoded_text = self.projectionEmbeddings(
            self.text_encoder(captions)
        )

        image_similarity = torch.matmul(encoded_images, torch.transpose(encoded_images))
        caption_similarity = torch.matmul(encoded_text, torch.transpose(encoded_text))
        targets = nn.Softmax((caption_similarity+image_similarity)/2)
        logits = torch.matmul(encoded_text, torch.transpose(encoded_images))

        return targets, logits