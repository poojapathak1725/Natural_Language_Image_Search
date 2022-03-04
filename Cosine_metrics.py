from dataset_factory import get_datasets
from file_utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from dataset_factory import get_datasets
from cnn_encoder import *
from Dataload import *
from pycocotools.coco import COCO




def cosine_similarity(img_id, encoded_text, loader):
    coco = COCO(json)
    path = coco.loadImgs(img_id)[0]['file_name'];
    image = Image.open(os.path.join(self.root, path)).convert('RGB')
    image_from_id = coco.anns[ann_id]['image_id']
    img_vector = EncoderCNN(image_from_id)
    cos = torch.nn.CosineSimilarity(dim=0)
    cosine_result = cos(img_vector, encoded_text)
    
    return cosine_result