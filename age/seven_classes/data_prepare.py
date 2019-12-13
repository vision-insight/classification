import os
import torch
import pathlib
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import PIL
from PIL import Image
from utils.torch_utils import *
from utils.utils import *
from utils.ImageFolder import ImageFolder
os.system("clear")


# 000  Assignment of variables ###################################

# Height and width of the CNN input image
img_h, img_w = 180,180

# Set train and valid directory paths
dataset_dir = "/data/lulei/data/age_data/data_split"
# Set the index and the corresponding class name (folder name) of the classes
idx_and_class = {0 : '0-6',
                 1 : '6-12',
                 2 : '12-18',
                 3 : '18-25', 
                 4 : '25-40',
                 5 : '40-55',
                 6 : '55+'}
# Batch size
batch_size = 512
print("[INFO] batch size : ", batch_size)


train_data_dir= os.path.join(dataset_dir, 'train')
valid_data_dir = os.path.join(dataset_dir, 'valid')
test_data_dir = os.path.join(dataset_dir, 'test')


# 001 Data Transforms #####################
image_transforms = { 
    # transforms (a.k.a data augmentations) for the training images
    'train': transforms.Compose([
        # transfer the input image into gray scale
        #transforms.Grayscale(num_output_channels=1),
        # resize the input image into the predefined scale
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC), # (h, w)
        # random choose one of the predefined transforms (in the list) when performing the training process
        #transforms.RandomChoice([transforms.RandomHorizontalFlip(),
        #                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)]),
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
    # transforms for the test images
    'test': transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
        #transforms.Resize((img_h, img_w), interpolation=PIL.Image.BICUBIC),
        transforms.Lambda(lambda img : pad_img(img, img_w)),
        transforms.ToTensor(),
    ])
}


# 002 Load Data from folders   ##################
data = {
    'train': ImageFolder(root=train_data_dir,
                                  idx_and_class = idx_and_class,
                                  transform=image_transforms['train'],
                                  ),

    'valid': ImageFolder(root=valid_data_dir, 
                                  idx_and_class = idx_and_class,
                                  transform=image_transforms['valid'],
                                  target_transform=None),


    'test': ImageFolder(root=test_data_dir, 
                                  idx_and_class = idx_and_class,
                                 transform=image_transforms['test'],
                                 target_transform=None)
}


# 003 Data iterators (Data loader) ###########################
train_data_loader = DataLoader(data['train'], 
                               batch_size=batch_size, 
                               shuffle=True,
                               num_workers=0)

valid_data_loader = DataLoader(data['valid'], 
                               batch_size=batch_size, 
                               shuffle=True,
                               num_workers=0)

test_data_loader = DataLoader(data['test'],
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=0)


# 004 get the weights of each classes ############
class_to_index = data["train"].class_to_idx
print("[INFO] class to index : ",class_to_index)

class_weights = get_class_weights(train_data_dir, idx_and_class)
print("[INFO] class weights : ", class_weights)

# 006 show the image quantity in each set
for data_type in ["train", "valid", "test"]:
    temp_1 = len(data[data_type])
    print("[INFO] image for %s : %d" % (data_type, temp_1))
    #for index in index_to_class.keys():
    #    temp_2 = sum([ 1 for info_tuple in data[data_type].imgs if info_tuple[1] == index])
    #    print("  | %s : %d(%.1f%%)" % (index_to_class[index], temp_2, temp_2/temp_1*100), end=" ")
    #print()

# 005  data visualization  ##############################
# for i, batch_data in enumerate(train_data_loader):
#         images, labels = batch_data
#         print(len(images), type(images[0]))
#         temp_img = transforms.ToPILImage()(images[0]).convert('RGB')
#         # print(temp_img.size)
#         temp_img.show()
        


if __name__ == "__main__":
    pass
