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

gc.collect()
torch.cuda.empty_cache()

class Training(object):

    def get_configs(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)
        return config_data

    def initialize(self, config_name):

        #Load Configs
        config_data = self.get_configs(config_name)

        #Initialize Data Loader
        _, _, self.vocabulary, self.train_loader, _, _ = get_datasets(config_data)

        # Setup Experiment
        self.generation_config = config_data['generation']
        self.epochs = config_data['experiment']['num_epochs']
        self.current_epoch = 0
        self.training_losses = []
        self.val_losses = []
        self.min_loss = float('inf')
        self.best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.dual_encoder = DualEncoder(config_data['dual_encoder_configs'])

        self.criterion = DualEncoderLoss()
        self.optimizer = optim.Adam(self.dual_encoder.parameters(), lr=config_data['learning_rate'])

        if torch.cuda.is_available():
            self.dual_encoder = self.dual_encoder.cuda().float()
            self.criterion = self.criterion.cuda()

    def train(self):

        for epoch in range(self.config_data['epochs']):
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
            
            torch.cuda.empty_cache()
      
        
        print("Finished epoch {}, time elapsed {}, Average Loss {}".format(
            epoch, 
            time.time() - start_time),
            epoch_loss / len(self.train_loader)
            )
    
def plot_val_train_loss(Training_Loss, Validation_Loss):

    Validation_Loss = Validation_Loss[1:]
    n = [i for i in range(1,len(Training_Loss)+1)]
    plt.plot(n, Training_Loss, label ="Training Loss")
    plt.plot(n, Validation_Loss, label ="Validation Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title("Train/Val Loss v.s. Epoch")
    plt.savefig('graph_1.png')
    plt.show()

if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
    plot_val_train_loss(Training_Loss, Validation_Loss)
    test()
    
    # housekeeping
    gc.collect() 
    torch.cuda.empty_cache()


# Load Text and Image Encoders


# Set Optimizers and Loss Function

# Implement Train Function

# Implement Validation Function

# Implementat Test Function
    # Extract Image IDs from stored directory