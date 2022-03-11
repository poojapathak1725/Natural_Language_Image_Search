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

import pdb

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
                
                targets, predictions = self.dual_encoder(images, captions)
               
                
                loss = self.criterion(predictions, targets)

                epoch_loss += loss.item()

                loss.backward()

                self.optimizer.step()

                if iter % 10 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                    
                if iter == 249:
                    break
            
            torch.cuda.empty_cache()
      
  
            print("Finished epoch {}, time elapsed {}, Average Loss {}".format(
                    epoch, 
                    (time.time() - start_time),
                    epoch_loss / 250
                )
            )
            training_loss.append(epoch_loss / 250)
            val_loss.append(self.val(epoch))
        self.test("Plate of food on a table")
#         self.best_text_path = os.path.join("./", 'best-text-model.pt')
#         self.best_image_path = os.path.join("./", 'best-image-model.pt')
        
#         torch.save(self.dual_encoder.text_encoder, self.best_text_path)
#         torch.save(self.dual_encoder.image_encoder, self.best_image_path)
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
                
                
                targets, predictions = self.dual_encoder(images, captions)
                loss = self.criterion(predictions, targets)
                
                val_loss += loss.item()
                
                if iter == 49:
                    break
        
        val_loss = val_loss / 50
                
        print("Validation Loss at Epoch {} is {}".format(epoch, val_loss))
              
        self.dual_encoder.train()
        
        gc.collect()
        torch.cuda.empty_cache() 
            
        return val_loss
           
    def tokenize_query(self, query):
        tokens = nltk.tokenize.word_tokenize(str(query).lower())
        caption = [self.vocabulary('<start>')]
        caption.extend([self.vocabulary(token) for token in tokens])
        caption.append(self.vocabulary('<end>'))
        return caption
        
    
    def find_images(self, image_embeddings, query, image_paths, k=7, normalize = True):
#         query = self.dual_encoder.text_encoder.src_tokenizer.encode(query)
#         tokenizer = self.dual_encoder.text_encoder.src_tokenizer
#         query = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))
        query = self.tokenize(query)
        query_embedding = self.dual_encoder.text_encoder(torch.as_tensor([query]).cuda())
        if normalize:
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        dot_similarity = torch.matmul(query_embedding, torch.transpose(image_embeddings, 0, 1))
        values, indices = torch.topk(dot_similarity,k)
        pdb.set_trace()
        indices = indices.cpu().detach().numpy()
        return [[image_paths[idx] for idx in x] for x in indices]

    def get_batch_image_paths(self, img_ids):
        image_path = "data_loading/data/images/test/COCO_val2014_"
        image_paths = [image_path + str(img_id).zfill(12) + ".jpg" for img_id in img_ids]
        return image_paths

    def load_model(self):
        self.dual_encoder.text_encoder = torch.load("0.001-best-text-model.pt")
        self.dual_encoder.image_encoder = torch.load("0.001-best-image-model.pt")
    
    def test(self, query):
        torch.cuda.empty_cache()
#         self.load_model()
        self.dual_encoder.eval()
        image_arrays = None
        with torch.no_grad():
            for iter, (images, _, img_ids) in enumerate(self.test_loader):
                if iter == 8:
                    break
                if torch.cuda.is_available():
                    images = images.cuda()
                if image_arrays == None:
                    image_arrays = images
                    image_ids = img_ids
                else:
                    image_arrays = torch.cat((image_arrays,images),0)
                    image_ids.extend(img_ids)
            image_paths = self.get_batch_image_paths(image_ids)
            image_embeddings = self.dual_encoder.image_encoder(image_arrays)
        matches = self.find_images(image_embeddings, query, image_paths, normalize=True)
        print(matches)

    
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
        plt.savefig('graph_1.png')
        plt.show()

if __name__ == "__main__":
    
    training_class = Training("config_data")
    training_loss, val_loss = training_class.train()
    training_class.plot_train_loss(training_loss, val_loss)
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()

