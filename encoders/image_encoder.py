import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch import add


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         last_layer = list(resnet.children())[:-1] 
#         self.resnet = nn.Sequential(*last_layer)
        
        self.resnet_50 = models.resnet50(pretrained=True)
            
        self.resnet_50.fc = nn.Linear(self.resnet_50.fc.in_features, 300)
        self.Gelu = nn.GELU()
        self.Dense_unit = nn.Linear(300, 300)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(300)
        
    def forward(self, images):
        resnet_outputs = self.resnet_50(images)
#         resnet_outputs = Variable(resnet_outputs.data)
#         resnet_outputs = resnet_outputs.view(resnet_outputs.size(0), -1)

        x = self.Gelu(resnet_outputs)
        x = self.Dense_unit(x)
        x = self.dropout(x)
        x = add(resnet_outputs, x)
        embed_proj = self.layer_norm(x)
        
        return embed_proj

