import os
import pathlib
import shutil
import random
from PIL import Image
import torch
import numpy as np


def head_center(image):
    iw, ih = image.size
    if ih >= iw*2:
        image = image.crop((0, 0, iw, ih//2))
    return image


def pad_img(image, long_side):
    '''
        rescale the input image to make its long side equal to long_side,
        and pad the rescaled image to form a square image with its side length equals to long_side
    '''
    iw, ih = image.size #
    w, h = (long_side, long_side)
    scale = min(w / iw, h / ih) #

    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.ANTIALIAS) #
    new_image = Image.new('RGB', (w, h), (128, 128, 128)) #

    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2)) #

    return new_image

def centralize(img,rw_ratio,lw_ratio,up_ratio,down_ratio):
    
    iw, ih = img.size
    center_x = iw / 2
    center_y = ih / 2
    rw = int(rw_ratio*iw)
    lw = int(lw_ratio*iw)
    up_h = int(up_ratio*ih)
    down_h = int(down_ratio*ih)

    new_img= img.crop(
    (
        center_x - lw ,
        center_y - up_h,
        center_x + rw,
        center_y + down_h
    )
    )
    return new_img

def verticalize(img):
    w, h = img.size
    if w > h:
        return img.rotate(90,expand = 1)
    else:
        return img

def get_class_weights(input_dir, idx_to_class, idx_first = True):
    '''
        get the weights of each classes accroding to the inverse ratio of th corresponding sample numbers
        input_dir : the root dir of the image classes, the folder structure under ipput_dir should be:
                    input_dir:
                            class_1:
                                    1.jpg
                                    2.jpg
                            class_2:
                                    1.jpg
                                    2.jpg
        class_to_index : a dict depicting the relationship between class_name and its index, should be in the
                         format of {class_name_1:0, classe_name_2:2,}. this could be obtained from .class_to_index
                         method of instance of ImageFolder
    '''

    #index_to_class = sorted([[v,k] for k, v in class_to_index.items()],key = lambda x: x[0])

    weights = []
    for key, value in idx_to_class.items():
        if idx_first:
            idx, class_name = key, value
        else:
            idx, class_name = value, key
        
        try:
            if isinstance(eval(class_name), list):
                class_name = eval(class_name)
                temp_sum = 0
                for subclass_name in class_name:
                    temp_num = len([i for i in pathlib.Path(os.path.join(input_dir, subclass_name)).rglob("*.jpg")])
                    temp_sum += temp_num
        except:
            temp_sum = len([i for i in pathlib.Path(os.path.join(input_dir, class_name)).rglob("*.jpg")])
        weights.append(1.0/temp_sum)
    weights = torch.tensor(weights).float()
    weights = weights/weights.sum()#*10

    return weights

def list_split(input_list, ratio = [0.5, 0.5], shuffle = False):
    assert sum(ratio) == 1, print("sum of ratio should be 1")
    input_list = sorted(input_list)
    len_list = len(input_list)

    split_len = []
    for i in ratio[:-1]:
        split_len.append(round(len_list*i))

    return_list = []
    temp = 0
    for i in split_len:
        start, end = temp, temp + i
        temp += i
        return_list.append(input_list[start: end])
    return_list.append(input_list[temp:])

    return return_list

def get_all_images(input_dir, ext = "jpg", shuffle = True):
    image_paths = [i for i in pathlib.Path(input_dir).rglob("*.jpg")] #(".".join(["*", ext]))]
    if shuffle:
        random.shuffle(image_paths)
    return image_paths

def imread(file_name, is_gray = False, resize = False, width = None, height = None):
    if is_gray:
        image = cv2.imread(str(file_name), flags = cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.cvtColor(cv2.imread(str(file_name)), cv2.COLOR_BGR2RGB)

    if resize:
        h, w = image.shape[:2]
        if width == None and height != None:
            width = int(height/h*w)
        elif width != None and height == None:
            height = int(width/w*width)

        image = cv2.resize(image, (width, height))

    return image

