################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import nltk
from PIL import Image
from pycocotools.coco import COCO
import torch.nn as nn
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json, ids, vocab, img_size, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformations.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = ids
        self.vocab = vocab
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.resize = transforms.Compose(
            [transforms.Resize(img_size, interpolation=2), transforms.CenterCrop(img_size)])
        
        self.transformations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomPerspective(p=0.5)
            
        ])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
#         import pdb; pdb.set_trace();
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = self.resize(image)
        image = self.normalize(np.asarray(image))
#         image = self.transformations(image)

        # Convert caption (string) to word ids.
#         tokens = nltk.tokenize.word_tokenize(str(caption).lower())
#         caption = [vocab(token) for token in tokens]

#         caption = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(caption))
#         output = tokenizer(caption, padding = True
#         target = torch.Tensor(caption)
        return image, caption, img_id

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption)
    by padding the captions to make them of equal length.
    We can not use default collate_fn because variable length tensors can't be stacked vertically.
    We need to pad the captions to make them of equal length so that they can be stacked for creating a mini-batch.
    Read this for more information - https://pytorch.org/docs/stable/data.html#dataloader-collate-fn
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, img_ids = zip(*data)
    images = torch.stack(images, 0)


#     # Merge captions (from tuple of 1D tensor to 2D tensor).

    output = tokenizer(captions, padding = True, truncation=True, return_tensors="pt")
    
    targets = output['input_ids']
    attention_masks = output['attention_mask']
    token_type_ids = output['token_type_ids']
                            
    
#     lengths = [len(cap) for cap in captions]
# #     targets = torch.zeros(len(captions), max(lengths)).long()

#     targets = torch.empty(len(captions), max(lengths)).fill_(tokenizer.pad_token_id).long()
# # #     import pdb; pdb.set_trace();
#     for i, cap in enumerate(captions):
#         end = lengths[i]
#         targets[i, :end] = cap[:end]
        
    return images, targets, attention_masks, token_type_ids, img_ids