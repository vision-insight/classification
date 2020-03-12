import os
import sys
import torch
import pathlib
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import PIL
from PIL import Image
from multiprocessing import cpu_count

base_path = "/data/lulei/classification"
sys.path.insert(0, base_path)
from tools.utils.torch_utils import *
from tools.utils.utils import *
from tools.utils.ImageFolder import ImageFolder
os.system("clear")


################## 00 variable  Assignment ################################

# Height and width of the CNN input image
img_h, img_w = 180,180

# Set train and valid directory paths
dataset_dir = "/data/lulei/data/age/version_1/data_split"

# Set the index and the corresponding class name (folder name) of the classes
idx_and_class = {0 : ['0-6', '6-12'],
                 1 : ['12-18', '18-25'],
                 2 : ['25-40', '40-55'],
                 3 : ['55+']}
# Batch size
batch_size = 256
print("[INFO] batch size : ", batch_size)


train_data_dir= os.path.join(dataset_dir, 'train')
valid_data_dir = os.path.join(dataset_dir, 'valid')
test_data_dir = os.path.join(dataset_dir, 'test')


########## 001 Data Transforms #####################

image_transforms = { 
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


############## 002 Load Data from folders   ##################

data = {
        'train': ImageFolder(root=train_data_dir,
                             idx_and_class = idx_and_class,
                             transform=image_transforms['train']),
        'valid': ImageFolder(root=valid_data_dir, 
                             idx_and_class = idx_and_class,
                             transform=image_transforms['valid'],
                             target_transform=None),
       }


############# 003 Data iterators (Data loader) ###########################

dataloaders = {
               "train": DataLoader(data['train'], 
                                   batch_size=batch_size, 
                                   shuffle=True,
                                   num_workers= cpu_count()//2),

               "valid": DataLoader(data['valid'], 
                                   batch_size=batch_size, 
                                   shuffle=True,
                                   num_workers= cpu_count()//2),
              }


############ 004 get the weights of each classes ############

class_to_index = data["train"].class_to_idx
print("[INFO] class to index : ",class_to_index)

class_weights = get_class_weights(train_data_dir, class_to_index, idx_first = False)
print("[INFO] class weights : ", class_weights)

############# 005 show the image quantity in each set ##########
for data_type in ["train", "valid"]:
    temp = len(data[data_type])
    print("[INFO] image for %s : %d" % (data_type, temp))
        


if __name__ == "__main__":
    pass
