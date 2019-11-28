import os
import shutil
import random
from PIL import Image

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
