from encoders.image_encoder import CNNEncoder
from encoders.text_encoder import LSTMEncoder, BERTModel
from encoders.projection_embedding import ImgTextEmbeddings
import torch.nn as nn
import torch
import torch.nn.functional as F

class DualEncoder(nn.Module):
    def __init__(self, configs, vocab_size) -> None:
        super().__init__()
        self.text_encoder = BERTModel()
        self.image_encoder = CNNEncoder()
        self.softmax = nn.Softmax(dim = -1)
        self.logit_scale = torch.ones([]) * 2.6592
        

    def forward(self, images, captions, attention_masks, token_type_ids):
        
        encoded_images = self.image_encoder(images)
        encoded_text = self.text_encoder(captions, attention_masks, token_type_ids)
        
        
        normalized_images = encoded_images / encoded_images.norm(dim=-1, keepdim=True)
        normalized_text = encoded_text / encoded_text.norm(dim=-1, keepdim=True)

#         normalized_images = encoded_images
#         normalized_text = encoded_text

        # cosine similarity as logits
        logits_per_text = torch.matmul(normalized_text, torch.transpose(normalized_images, 0, 1)) * self.logit_scale.exp()
        logits_per_image = torch.transpose(logits_per_text, 0, 1)
        
                                        
        return logits_per_text, logits_per_image
#         return normalized_images, normalized_text
        