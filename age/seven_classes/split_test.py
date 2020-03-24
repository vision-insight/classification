import os
import sys
import torch
import pathlib
from torch.utils.data import Dataset, DataLoader, random_split
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

# Height and width of the CNN input image
img_h, img_w = 227,400

# Set train and valid directory paths
dataset_dir = "/media/D/lulei/data/age/origin"

# Batch size
batch_size = 20
print("[INFO] batch size : ", batch_size)


########## 001 Data Transforms #####################

image_trans = { 
    'train': transforms.Compose([
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        transforms.ToTensor(),
                                ]),

    'valid': transforms.Compose([
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        transforms.ToTensor(),
                                ]),
             }

############## 002 Load Data from folders   ##################
origin_data = ImageFolder(root = dataset_dir)
data = {}
data["train"], data["valid"] = random_split(origin_data,
                                     (int(len(origin_data)*0.7), len(origin_data) - int(len(origin_data)*0.7)))

data["train"].dataset.transform = image_trans['train']
data["valid"].dataset.transform = image_trans['valid']                        


############# 003 Data iterators (Data loader) ###########################
print(data["train"].dataset.imgs[100])
print(data["train"].dataset.imgs[200])

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

class_to_index = data["train"].dataset.class_to_idx
index_to_class = { v:k for k,v in class_to_index.items()}
if len(class_to_index) <= 20: print("[INFO] class to index : ",class_to_index)

class_weights = get_class_weights(dataset_dir, class_to_index, idx_first = False)*10
print("[INFO] class weights : ", class_weights)

############# 005 show the image quantity in each set ##########
for data_type in ["train", "valid"]:
    print(f"[INFO] image for {data_type} : {len(data[data_type])}")
        
