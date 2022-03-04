import torch
import torch.nn as nn

def dual_encoder_loss(predictions, targets):
    cross_entropy = nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean")
    return (cross_entropy(predictions) + cross_entropy(target))/2

# place holders for image and text encodings got from encoders
image_embedding = torch.Tensor([])
text_embedding = torch.Tensor([])

# calculating target values
image_similarity = torch.matmul(image_embedding, torch.transpose(image_embedding))
caption_similarity = torch.matmul(text_embedding, torch.transpose(text_embedding))
target = nn.Softmax((caption_similarity+image_similarity)/2)

# computing logits
logits = torch.matmul(text_embedding, torch.transpose(image_embedding))

# defining the loss function
criterion = dual_encoder_loss

loss = criterion(logits, target)
print("Loss:", loss.item())