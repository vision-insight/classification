from torch.utils.data import Dataset, DataLoader, random_split^M
from torchvision import datasets, models, transforms^M
import PIL^M
from PIL import Image^M
from multiprocessing import cpu_count^M
import sys

base_path = "/media/D/lulei/classification"^M
sys.path.insert(0, base_path)^M
from tools.utils.torch_utils import *^M
from tools.utils.torch_utils import *^M
from tools.utils.utils import *^M
from tools.utils.sampler import BalancedBatchSampler^M
from tools.utils.ImageFolder import ImageFolder^M
os.system("clear")^M
^M
################## 00 variables^M
^M
img_h, img_w = 227, 400^M
^M
n_classes = 4^M
^M
train_ratio = 0.9^M
^M
dataset_dir = "/media/D/lulei/data/age/origin"^M
^M
batch_size = 60^M
print("[INFO] batch size : ", batch_size)^M
^M
# Set the index and the corresponding class name (folder name) of the classes^M
idx_and_class = {0 : ['0-8', '8-18'],^M
                 1 : ['18-25', '25-35'],^M
                 2 : ['35-45', '45-65'],^M
                 3 : ['65+']}^M
^M
^M
########## 001 Data Transforms #####################^M
^M
image_trans = { ^M
    # transforms (a.k.a data augmentations) for the training images^M
    'train': transforms.Compose([^M
        # transfer the input image into gray scale^M
        transforms.Lambda(lambda img : pad_img(img, img_w)),^M
        # transfer the type of input image into tensor style^M
        transforms.ToTensor(),^M
                                ]),^M
^M
    # transforms for the valid images^M
    'valid': transforms.Compose([^M
        #transforms.Grayscale(num_output_channels=1),^M
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC),^M
        transforms.Lambda(lambda img : pad_img(img, img_w)),^M
        transforms.ToTensor(),^M
                                ]),^M
                  }^M
^M
^M
############## 002 Load Data from folders^M
^M
origin_data = ImageFolder(root = dataset_dir, idx_and_class = idx_and_class)^M
^M
train_num = int(len(origin_data) * train_ratio)^M
valid_num = len(origin_data) - train_num^M
^M
data = {}^M
data["train"], data["valid"] = random_split(origin_data, (train_num, valid_num))^M
^M
^M
^M
data["train"].dataset.transform = image_trans['train']^M
data["valid"].dataset.transform = image_trans['valid']^M
^M
#balanced_batch_sampler = BalancedBatchSampler(data["train"], n_classes = 4, n_samples = 20, batch_num = 80)^M
^M
#balanced_batch_sampler = BalancedBatchSampler(data["train"])^M
^M
for i in ["train", "valid"]:^M
        print(f"[INFO] {i} data num : {len(data[i])}")^M
############# 003 Data iterators (Data loader) ###########################^M
^M
dataloaders = {^M
            "train": DataLoader(data["train"],^M
                                batch_size=batch_size,^M
#                                sampler = balanced_batch_sampler,^M
                                num_workers= cpu_count()),^M
^M
                "valid": DataLoader(data["valid"],^M
                                    batch_size=batch_size,^M
                                    shuffle=True,^M
                                    num_workers= cpu_count()),^M
                              }^M
^M
############ 004 get the weights of each classes ############^M
class_to_index = data["train"].dataset.class_to_idx^M
index_to_class = { v:k for k,v in class_to_index.items()}^M
if n_classes < 10: print(f"[INFO] class to index : {data['train'].dataset.class_to_idx}")^M
^M
class_weights = get_class_weights(dataset_dir, data["train"].dataset.class_to_idx, idx_first = False)^M
print(f"[INFO] class weights : {class_weights}")^
                                                           
