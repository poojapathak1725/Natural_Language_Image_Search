import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    This criterion (`CrossEntropyLoss`) combines `LogSoftMax` and `NLLLoss` in one single class.
    
    NOTE: Computes per-element losses for a mini-batch (instead of the average loss over the entire mini-batch).
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax()
    
    def forward(self, pred, target):
        loss = - torch.sum(target * self.log_softmax(pred))
        return loss / float(pred.shape[0])


class DualEncoderLoss(nn.Module):
    def __init__(self):
        super(DualEncoderLoss, self).__init__()
        self.cross_entropy = CrossEntropyLoss()
        
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def forward(self,text, image):
        
        loss_val = self.clip_loss(text)
        
        return loss_val


        

    

