import torch
import torch.nn as nn

class DualEncoderLoss(nn.Module):
    def __init__(self):
        super(DualEncoderLoss, self).__init__()
#         self.cross_entropy = nn.KLDivLoss(size_average=None, ignore_index=-100, reduce=None, reduction="mean")
#         self.cross_entropy = nn.CrossEntropyLoss()
        self.cross_entropy = nn.MSELoss()

    def forward(self,predictions,targets):
#         import pdb; pdb.set_trace();
        return (
            self.cross_entropy(predictions,targets) + 
            self.cross_entropy(torch.transpose(predictions, 0, 1), torch.transpose(targets, 0, 1))
        ) / 2

    
