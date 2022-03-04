from dataset_factory import get_datasets
from file_utils import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from dataset_factory import get_datasets

class Experiment():
    def __init__(self,name):
        ROOT_STATS_DIR = './experiment_data'
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_train, self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)
        
    def train(self):
        print("COCO TEST WORKS ---- ",self.__coco_train)
        reference_obj = self.__coco_train
        total_captions = []
        total_images = []
        for iter, (images, captions, img_ids) in enumerate(self.__train_loader):
            images = images.cuda()
            captions = captions.cuda()
            
            import pdb; pdb.set_trace();
            
            for each_img_id in range(len(img_ids)):
                caption_each_img = []
                image_for_img_id = [images[each_img_id] for _ in range(len(reference_obj.imgToAnns[img_ids[each_img_id]]))]
                for annotation in reference_obj.imgToAnns[img_ids[each_img_id]]:
                    
                    caption = str(annotation['caption']).lower()
                    
                    caption_each_img.append(caption)
                total_captions.append(caption_each_img)
                total_images.append(image_for_img_id)
            import pdb; pdb.set_trace();
            
            
            
            
    def val(self):
        print("COCO TEST WORKS ---- ",self.__coco_test)
        reference_obj = self.__coco_test
        total_captions = []
        total_images = []
        for iter, (images, captions, img_ids) in enumerate(self.__val_loader):
            images = images.cuda()
            captions = captions.cuda()
            
            import pdb; pdb.set_trace();
            
            for each_img_id in range(len(img_ids)):
                caption_each_img = []
                image_for_img_id = [images[each_img_id] for _ in range(len(reference_obj.imgToAnns[img_ids[each_img_id]]))]
                for annotation in reference_obj.imgToAnns[img_ids[each_img_id]]:
                    
                    caption = str(annotation['caption']).lower()
                    
                    caption_each_img.append(caption)
                total_captions.append(caption_each_img)
                total_images.append(image_for_img_id)
            import pdb; pdb.set_trace();
             
                
                
                
                
            
    def test(self):
        print("COCO TEST WORKS ---- ",self.__coco_test)
        reference_obj = self.__coco_test
        total_captions = []
        total_images = []
        for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
            images = images.cuda()
            captions = captions.cuda()
            
            import pdb; pdb.set_trace();
            
            for each_img_id in range(len(img_ids)):
                caption_each_img = []
                image_for_img_id = [images[each_img_id] for _ in range(len(reference_obj.imgToAnns[img_ids[each_img_id]]))]
                for annotation in reference_obj.imgToAnns[img_ids[each_img_id]]:
                    
                    caption = str(annotation['caption']).lower()
                    
                    caption_each_img.append(caption)
                total_captions.append(caption_each_img)
                total_images.append(image_for_img_id)
            import pdb; pdb.set_trace();
             
                
        
exp = Experiment('default')
exp.train()