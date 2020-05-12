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
from classification.tools.utils.torch_utils import *
from tools.utils.utils import *
from tools.utils.sampler import BalancedBatchSampler
from tools.utils.ImageFolder import ImageFolder
os.system("clear")



################## 00 variables

img_h, img_w = 227, 400

n_classes = 4

train_ratio = 0.9

dataset_dir = "/media/D/lulei/data/age/origin"

batch_size = 60
print("[INFO] batch size : ", batch_size)

# Set the index and the corresponding class name (folder name) of the classes
idx_and_class = {0 : ['0-8', '8-18'],
                 1 : ['18-25', '25-35'],
                 2 : ['35-45', '45-65'],
                 3 : ['65+']}


########## 001 Data Transforms #####################

image_trans = { 
    # transforms (a.k.a data augmentations) for the training images
    'train': transforms.Compose([
        # transfer the input image into gray scale
        #transforms.Grayscale(num_output_channels=1),
        # resize the input image into the predefined scale
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC), # (h, w)
        # random choose one of the predefined transforms (in the list) when performing the training process
        #transforms.RandomChoice([transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)]),
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        # transfer the type of input image into tensor style
        transforms.ToTensor(),
                                ]),

    # transforms for the valid images
    'valid': transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC),
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        transforms.ToTensor(),
                                ]),
                  }


############## 002 Load Data from folders

origin_data = ImageFolder(root = dataset_dir, idx_and_class = idx_and_class)

train_num = int(len(origin_data) * train_ratio)
valid_num = len(origin_data) - train_num

data = {}
data["train"], data["valid"] = random_split(origin_data, (train_num, valid_num))



data["train"].dataset.transform = image_trans['train']
data["valid"].dataset.transform = image_trans['valid']

#balanced_batch_sampler = BalancedBatchSampler(data["train"], n_classes = 4, n_samples = 20, batch_num = 80)

#balanced_batch_sampler = BalancedBatchSampler(data["train"])

for i in ["train", "valid"]:
        print(f"[INFO] {i} data num : {len(data[i])}")

############# 003 Data iterators (Data loader) ###########################

dataloaders = {
            "train": DataLoader(data["train"],
                                batch_size=batch_size,
#                                sampler = balanced_batch_sampler,
                                num_workers= cpu_count()),

                "valid": DataLoader(data["valid"],
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers= cpu_count()),
                              }

############ 004 get the weights of each classes ############
class_to_index = data["train"].dataset.class_to_idx
index_to_class = { v:k for k,v in class_to_index.items()}
if n_classes < 10: print(f"[INFO] class to index : {data['train'].dataset.class_to_idx}")

class_weights = get_class_weights(dataset_dir, data["train"].dataset.class_to_idx, idx_first = False)
print(f"[INFO] class weights : {class_weights}")
