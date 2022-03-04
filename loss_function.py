import torch
import torch.nn as nn

class DualEncoderLoss(nn.Module):
    def __init__(self):
        super(DualEncoderLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean")
    
    def forward(self,predictions,targets):
        return (self.cross_entropy(predictions,targets) + self.cross_entropy(torch.transpose(predictions), torch.transpose(targets))) / 2

    
