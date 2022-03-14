from distutils.command.config import config
import time

from matplotlib import image
from data_loading.coco_dataset import CocoDataset
from data_loading.dataset_factory import get_datasets
from file_utils import *
from dual_encoder import DualEncoder
from loss_function import DualEncoderLoss
import torch.optim as optim
import torch
import gc
import matplotlib.pyplot as plt
import os
import nltk
import numpy as np
import pdb
from PIL import Image
import torch.nn as nn

gc.collect()
torch.cuda.empty_cache()

class Training(object):

    def get_configs(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        return config_data

    def __init__(self, config_name):

        #Load Configs
        self.config_data = self.get_configs(config_name)

        #Initialize Data Loader
        _, _, self.vocabulary, self.train_loader, self.val_loader, self.test_loader, self.test_ids_file_path = get_datasets(self.config_data)

        # Setup Experiment
        self.epochs = self.config_data['epochs']
        self.current_epoch = 0
        self.training_losses = []
        self.val_losses = []
        self.min_loss = float('inf')
        self.best_model = None  # Save your best model in this field and use this in test method.
        self.config_data['dual_encoder_configs']['vocab_size'] = len(self.vocabulary)

        # Init Model
        self.dual_encoder = DualEncoder(self.config_data['dual_encoder_configs'], len(self.vocabulary))

        self.criterion = DualEncoderLoss()
        self.optimizer = optim.Adam(self.dual_encoder.parameters(), lr=self.config_data['learning_rate'])

        if torch.cuda.is_available():
            self.dual_encoder = self.dual_encoder.cuda().float()
            self.criterion = self.criterion.cuda()


    # Copied from transformers.models.clip.modeling_clip.clip_loss
    def clip_loss(self, text_similarity: torch.Tensor, image_similarity: torch.Tensor):
        caption_loss = nn.functional.cross_entropy(text_similarity, torch.arange(len(text_similarity), device=text_similarity.device))
        image_loss = nn.functional.cross_entropy(image_similarity, torch.arange(len(image_similarity), device=image_similarity.device))
        
        return (caption_loss + image_loss) / 2.0

    def train(self):
        
        training_loss = []
        val_loss = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            start_time = time.time()
            
            for iter, (images, captions, img_ids) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                if torch.cuda.is_available():
                    images = images.cuda()
                    captions = captions.cuda()
                
                text_pred, image_pred = self.dual_encoder(images, captions)
                
                loss = self.criterion(text_pred, image_pred)
                epoch_loss += loss.item()

                loss.backward()
                self.optimizer.step()
    
                if iter == 899:
                    break

                if iter % 10 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
            
                torch.cuda.empty_cache()
                gc.collect()
      
  
            print("Finished epoch {}, time elapsed {}, Average Loss {}".format(
                    epoch, 
                    (time.time() - start_time),
                    epoch_loss / 900
#                 len(self.train_loader)
                )
            )
            training_loss.append(epoch_loss / 900)
#                                  len(self.train_loader))
            val_loss.append(self.val(epoch))
            
        self.best_text_path = os.path.join("./", 'best-text-model-bert.pt')
        self.best_image_path = os.path.join("./", 'best-image-model-bert.pt')
        
        torch.save(self.dual_encoder.text_encoder, self.best_text_path)
        torch.save(self.dual_encoder.image_encoder, self.best_image_path)
        return training_loss, val_loss
    
    def val(self, epoch):
        torch.cuda.empty_cache() 
        self.dual_encoder.eval() # Put in eval mode (disables batchnorm/dropout) !

        val_loss = 0
        
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.val_loader): 

                if torch.cuda.is_available():
                    images = images.cuda()
                    captions = captions.cuda()
                
                logits_per_text, logits_per_image = self.dual_encoder(images, captions)
                loss = self.criterion(logits_per_text, logits_per_image)
                
                val_loss += loss.item()
        
        val_loss = val_loss / len(self.val_loader)
                
        print("Validation Loss at Epoch {} is {}".format(epoch, val_loss))
              
        self.dual_encoder.train()
        
        gc.collect()
        torch.cuda.empty_cache() 
            
        return val_loss
    
    def bert_tokenize(self, query):
        caption = self.dual_encoder.text_encoder.src_tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))
        return caption
           
    def tokenize_query(self, query):
        tokens = nltk.tokenize.word_tokenize(str(query).lower())
        caption = [self.vocabulary(token) for token in tokens]
        return caption
        
    
    def find_images(self, image_embeddings, query, image_paths, image_array, k=7, normalize = True):
        
        captions = self.bert_tokenize(query)
        
        if torch.cuda.is_available():
            captions = captions.cuda()

        query_embedding = self.dual_encoder.text_encoder(captions)
        
        
        if normalize:
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        dot_similarity = torch.matmul(query_embedding, torch.transpose(image_embeddings, 0, 1))
        values, indices = torch.topk(dot_similarity,k)
        
#         pdb.set_trace()
        indices = indices.cpu().detach().numpy()
        
        root = os.path.join("./", "answers/")
        
        for idx in indices[0]:
            img = image_array[idx].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
            img *= 255.0
            im = Image.fromarray(img.astype(np.uint8))
#             im.save(os.path.join(root, str(idx) + ".jpeg"))
            im.save(str(idx) + ".jpeg")
            
        return [[image_paths[idx] for idx in x] for x in indices]

    def get_batch_image_paths(self, img_ids):
        image_path = "data_loading/data/images/test/COCO_val2014_"
        image_paths = [image_path + str(img_id).zfill(12) + ".jpg" for img_id in img_ids]
        return image_paths

    def load_model(self):
        self.dual_encoder.text_encoder = torch.load("best-text-model-bert.pt")
        self.dual_encoder.image_encoder = torch.load("best-image-model-bert.pt")
    
    def test(self, query):
        torch.cuda.empty_cache()
        self.load_model()
        self.dual_encoder.eval()
        self.image_array = None
        image_embeddings = None
        with torch.no_grad():
            for iter, (images, _, img_ids) in enumerate(self.test_loader):
                if iter == 60: break
                print("Embedding batch",iter)
                if torch.cuda.is_available():
                    images = images.cuda()
                embeddings = self.dual_encoder.image_encoder(images)
                if image_embeddings == None:
                    image_embeddings = embeddings
                    self.image_array = images
                    image_ids = img_ids
                else:
                    image_embeddings = torch.cat((image_embeddings,embeddings),0)
                    self.image_array = torch.cat((self.image_array,images),0)
                    image_ids.extend(img_ids)
            image_paths = self.get_batch_image_paths(image_ids)
        matches = self.find_images(image_embeddings, query, image_paths, self.image_array, normalize=True)

    
#     def test(self, query):
#         torch.cuda.empty_cache()
#         self.load_model()
#         self.dual_encoder.eval()
#         image_arrays = None
#         with torch.no_grad():
#             for iter, (images, captions, attention_masks, token_type_ids, img_ids) in enumerate(self.test_loader):
#                 if iter == 8:
#                     break
#                 if torch.cuda.is_available():
#                     images = images.cuda()
#                 if image_arrays == None:
#                     image_arrays = images
#                     image_ids = img_ids
#                 else:
#                     image_arrays = torch.cat((image_arrays,images),0)
#                     image_ids.extend(img_ids)
#             image_paths = self.get_batch_image_paths(image_ids)
#             image_embeddings = self.dual_encoder.image_encoder(image_arrays)
#         matches = self.find_images(image_embeddings, query, image_paths, image_arrays, normalize=True)
    #         print(matches)

    
    def plot_train_loss(self, Training_Loss, Validation_Loss):
        
        write_to_file_in_dir("./", "train.txt", Training_Loss)
                
        write_to_file_in_dir("./", "val.txt", Validation_Loss)

        n = [i for i in range(1,len(Training_Loss)+1)]
        plt.plot(n, Training_Loss, label ="Training Loss")
        plt.plot(n, Validation_Loss, label ="Validation Loss")
        
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title("Train and Val Loss v.s. Epochs")
        plt.savefig('graph_2.png')
        plt.show()

if __name__ == "__main__":
    
    training_class = Training("config_data")
    training_loss, val_loss = training_class.train()
    training_class.plot_train_loss(training_loss, val_loss)

    training_class.test("a man walking on the beach with his surfboard")
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()
