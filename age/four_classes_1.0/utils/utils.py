import os
import pathlib
import shutil
import random
from PIL import Image
import torch

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



def get_class_weights(input_dir, class_to_index):
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
    index_to_class = sorted([[v,k] for k, v in class_to_index.items()],key = lambda x: x[0])
    weights = []
    for index, class_name in index_to_class:
        temp_num = len([i for i in pathlib.Path(os.path.join(input_dir, class_name)).rglob("*.jpg")])
        weights.append(1.0/temp_num)
    weights = torch.tensor(weights).float()
    weights = weights/weights.sum()#*10

    return weights
