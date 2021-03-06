import os
import sys
import torch
import pathlib
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import PIL
from PIL import Image
from multiprocessing import cpu_count

base_path = "/media/D/lulei/classification"
sys.path.insert(0, base_path)
from tools.utils.torch_utils import *
from tools.utils.utils import *
from torchvision.datasets import ImageFolder
os.system("clear")


################## 00 variable  Assignment ################################

img_h, img_w = 260, 140

n_classes = 2

dataset_dir = "/media/D/lulei/data/bald/split"

batch_size = 50
print("[INFO] batch size : ", batch_size)

train_data_dir= os.path.join(dataset_dir, 'train')
valid_data_dir = os.path.join(dataset_dir, 'valid')

########## 001 Data Transforms #####################

image_trans = { 
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.RandomRotation(30, resample=False, expand=True, center=None),
            transforms.Lambda(lambda image : centralize(image, 0,0,0,0.1)),
                                ]),

        transforms.Lambda(lambda img : pad_img(img, img_w)),
        transforms.ToTensor(),
                                ]),

    'valid': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC),
        #transforms.Lambda(lambda img : centralize(img,0.4,0.4,0.4,0.3)),
        #transforms.Lambda(lambda img : head_center(img)),
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        #transforms.RandomChoice([
            #transforms.Grayscale(num_output_channels=3),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            #transforms.Lambda(lambda img : centralize(img,0.4,0.4,0.5,0.5)),
            #transforms.RandomRotation(10, resample=False, expand=False, center=None)
                                #]),
        transforms.ToTensor(),
                                ]),

             }

############## 002 Load Data from folders   ##################

data = {
    'train': ImageFolder(root=train_data_dir,
                         transform=image_trans['train'], ),

    'valid': ImageFolder(root=valid_data_dir, 
                         transform=image_trans['valid'],
                         target_transform=None),
        }


############# 003 Data iterators (Data loader) ###########################

dataloaders = {
    "train": DataLoader(data['train'], 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers= cpu_count()),

    "valid": DataLoader(data['valid'], 
                        batch_size=batch_size*2, 
                        shuffle=True,
                        num_workers= cpu_count()),

              }

############ 004 get the weights of each classes ############

class_to_index = data["train"].class_to_idx
index_to_class = { v:k for k,v in class_to_index.items()}
if len(class_to_index) <= 20: print("[INFO] class to index : ",class_to_index)

class_weights = get_class_weights(train_data_dir, class_to_index, idx_first = False)
print("[INFO] class weights : ", class_weights)

############# 005 show the image quantity in each set ##########
for data_type in ["train", "valid"]:
    print(f"[INFO] image for {data_type} : {len(data[data_type])}")
        

if __name__ == "__main__":
    pass
