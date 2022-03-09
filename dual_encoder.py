from encoders.image_encoder import CNNEncoder
from encoders.text_encoder import LSTMEncoder
from encoders.projection_embedding import ImgTextEmbeddings
import torch.nn as nn
import torch
import torch.nn.functional as F

class DualEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.text_encoder = LSTMEncoder(configs['vocab_size'])
        self.image_encoder = CNNEncoder()

    def forward(self, images, captions):
        
        encoded_images = self.image_encoder(images)
        encoded_text = self.text_encoder(captions)
        
#         import pdb; pdb.set_trace();

        image_similarity = torch.matmul(encoded_images, torch.transpose(encoded_images, 0, 1))
        
        caption_similarity = torch.matmul(encoded_text, torch.transpose(encoded_text, 0, 1))
        
        targets = (caption_similarity+image_similarity)/2
        
        logits = torch.matmul(encoded_text, torch.transpose(encoded_images, 0, 1))

        return targets, logits