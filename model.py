import torch.nn as nn
from torchvision import models
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        last_layer = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*last_layer)
        self.initialize_weights()
        
    def forward(self, images):
        resnet_outputs = self.resnet(images)
        resnet_outputs = Variable(resnet_outputs.data)
        resnet_outputs = resnet_outputs.view(resnet_outputs.size(0), -1)
        return resnet_outputs