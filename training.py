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
        _, _, self.vocabulary, self.train_loader, self.val_loader, _ = get_datasets(self.config_data)

        # Setup Experiment
        self.epochs = self.config_data['epochs']
        self.current_epoch = 0
        self.training_losses = []
        self.val_losses = []
        self.min_loss = float('inf')
        self.best_model = None  # Save your best model in this field and use this in test method.
        self.config_data['dual_encoder_configs']['vocab_size'] = len(self.vocabulary)

        # Init Model
        self.dual_encoder = DualEncoder(self.config_data['dual_encoder_configs'])

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
            
#             val_loss.append(self.val(epoch))
            
        return training_loss
    
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
           

    
    def plot_train_loss(self, Training_Loss):
        
        write_to_file_in_dir("./", "train.txt", Training_Loss)
                
#         write_to_file_in_dir("./", "val.txt", Validation_Loss)

        n = [i for i in range(1,len(Training_Loss)+1)]
        plt.plot(n, Training_Loss, label ="Training Loss")
#         plt.plot(n, Validation_Loss, label ="Validation Loss")
        
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.title("Train Loss v.s. Epochs")
        plt.savefig('graph_1.png')
        plt.show()

if __name__ == "__main__":
    
    training_class = Training("config_data")
    training_loss = training_class.train()
    training_class.plot_train_loss(training_loss)
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()

