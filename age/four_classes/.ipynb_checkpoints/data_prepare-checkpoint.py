from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms
import PIL
from PIL import Image
from multiprocessing import cpu_count
import sys

base_path = "/data/lulei/classification"
sys.path.insert(0, base_path)
from tools.utils.torch_utils import *
from tools.utils.torch_utils import *
from tools.utils.utils import *
from tools.utils.sampler import BalancedBatchSampler
from tools.utils.ImageFolder import ImageFolder
os.system("clear")

def c_crop(image):
    w, h = image.size #
    if h > w*2:
        image = image.crop((0, 0, w, w))
        return image
    else:
        return image
################## 00 variables

img_h, img_w = 227, 200

n_classes = 4

dataset_dir = "/data/lulei/data/age/version_2/split/"

batch_size = 180
print("[INFO] batch size : ", batch_size)

train_data_dir= os.path.join(dataset_dir, 'train')
valid_data_dir = os.path.join(dataset_dir, 'valid')

idx_and_class = {0 : ['0-8', '8-18'],
                 1 : ['18-25', '25-35'],
                 2 : ['35-45', '45-65'],
                 3 : ['65+']}

########## 001 Data Transforms #####################^M
image_trans = { 
    # transforms (a.k.a data augmentations) for the training images
    'train': transforms.Compose([
        # transfer the input image into gray scale
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(lambda img : c_crop(img)),
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        # transfer the type of input image into tensor style
        transforms.ToTensor(),
                                ]),

    # transforms for the valid images
    'valid': transforms.Compose([
        transforms.Lambda(lambda img : c_crop(img)),
        transforms.Grayscale(num_output_channels=1),
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC),
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        transforms.ToTensor(),
                                ]),
                  }


############## 002 Load Data from folders   ##################

data = {
    'train': ImageFolder(root=train_data_dir,
                         transform=image_trans['train'],
                         idx_and_class = idx_and_class,
                         ),

    'valid': ImageFolder(root=valid_data_dir,
                         transform=image_trans['valid'],
                         idx_and_class = idx_and_class,
                         ),
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
