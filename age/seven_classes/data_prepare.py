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


################## 00 variables

img_h, img_w = 227, 400

class_num = 7

train_ratio = 0.9

dataset_dir = "/media/D/lulei/data/age/origin"

batch_size = 20
print("[INFO] batch size : ", batch_size)

################### 01 Data Transforms

image_trans = { 
    'train': transforms.Compose([
        # transfer the input image into gray scale
        #transforms.Grayscale(num_output_channels=1),
        # resize the input image into the predefined scale
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC), # (h, w)
        # random choose one of the predefined transforms (in the list) when performing the training process
        #transforms.Lambda(lambda img : head_center(img)),
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        #transforms.RandomChoice([
        #    transforms.RandomHorizontalFlip(),
        #    #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        #    #transforms.Lambda(lambda img : centralize(img,0.4,0.4,0.5,0.5)),
        #    transforms.RandomRotation(30, resample=False, expand=False, center=None)
        #                        ]),
        
        #transforms.Lambda(lambda img : verticalize(img)),
        # transfer the type of input image into tensor style
        transforms.ToTensor(),
                                ]),

    'valid': transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC),
        #transforms.Lambda(lambda img : centralize(img,0.4,0.4,0.4,0.3)),
        #transforms.Lambda(lambda img : head_center(img)),
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        #transforms.RandomChoice([
        #    transforms.RandomHorizontalFlip(),
        #    #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        #    #transforms.Lambda(lambda img : centralize(img,0.4,0.4,0.5,0.5)),
        #    transforms.RandomRotation(30, resample=False, expand=False, center=None)
        #                        ]),
        transforms.ToTensor(),
                                ]),

             }

############## 002 Load Data from folders

origin_data = ImageFolder(root = dataset_dir)

train_num = int(len(origin_data) * train_ratio)
valid_num = len(origin_data) - train_num

data = {}
data["train"], data["valid"] = random_split(origin_data, (train_num, valid_num))
#
#index_set = {}
#for index, (_ , label) in enumerate(data["train"]):
#    if label not in index_set.keys():
#        index_set.update({label:[]})
#    else:
#        index_set[label].append(index)
#
#print(index_set[0][:100])

data["train"].dataset.transform = image_trans['train']
data["valid"].dataset.transform = image_trans['valid']                        


balanced_batch_sampler = BalancedBatchSampler(data["train"], n_classes = 4, n_samples = 20, batch_num = 80)





for i in ["train", "valid"]:
    print(f"[INFO] {i} data num : {len(data[i])}")
############# 003 Data iterators (Data loader) ###########################

dataloaders = {
    "train": DataLoader(data["train"], 
                        batch_size=batch_size, 
                        #shuffle=True,
                        sampler = balanced_batch_sampler,
                        num_workers= cpu_count()),

    "valid": DataLoader(data["valid"], 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers= cpu_count()),
              }

############ 004 get the weights of each classes ############

if class_num < 10: print(f"[INFO] class to index : {data['train'].dataset.class_to_idx}")

class_weights = get_class_weights(dataset_dir, data["train"].dataset.class_to_idx, idx_first = False)
print(f"[INFO] class weights : {class_weights}")

