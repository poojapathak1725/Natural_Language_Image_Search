import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
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

    def forward(self,predictions,targets):
#         predictions = torch.log(predictions)
#         targets = torch.log(targets)
        
#         import pdb; pdb.set_trace();
        return (
            self.cross_entropy(predictions,targets) + 
            self.cross_entropy(torch.transpose(predictions, 0, 1), torch.transpose(targets, 0, 1))
        ) / 2

    

