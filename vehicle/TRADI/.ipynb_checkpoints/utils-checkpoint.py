import cv2
import numpy as np
from matplotlib import pyplot as plt



def region_split(image, d=[0.205, 0.205, 0.16, 0.205, 0.205 ] ):
    '''
        split the ROI into 5 region, they are (from left to right):
        ligths, joint parts, grill, logo region, grill, joints parts, lights
        with the corresponding ratio list above.
    '''
    
    n_r, n_c = image.shape
    
    pad = 0.01
    
    c_ll=[0 , round((d[0] + pad)* n_c)]
    c_lg=[round( (d[0])*n_c ) ,  round( (d[0]+d[1]+pad)*n_c )  ]
    c_l= [round( (d[0]+d[1])*n_c ) ,round( (d[0]+d[1]+d[2])*n_c ) ]
    c_rg=[round( (1-d[3]-d[4]-pad)*n_c), round( (1-d[4])*n_c)]
    c_rl=[round( (1-d[4]-pad)*n_c) ]
    
    
    ll = image[:,c_ll[0]:c_ll[1]]
    lg = image[:,c_lg[0]:c_lg[1]]
    logo = image[:,c_l[0]:c_l[1]]
    rg= image[:,c_rg[0]:c_rg[1]][:,-1::-1]
    rl= image[:,c_rl[0]:][:,-1::-1]

    return [ll, lg, logo, rg, rl]


def imread(file_name, is_gray = False, resize = False, width = None, height = None):
    if is_gray:
        image = cv2.imread(str(file_name), flags = cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.cvtColor(cv2.imread(str(file_name)), cv2.COLOR_BGR2RGB)
    print(image.shape)
    
    if resize:
        h, w = image.shape[:2]
        if width == None and height != None:
            width = int(height/h*w)
        elif width != None and height == None:
            height = int(width/w*width)

        image = cv2.resize(image, (width, height), 2)
        
    return image


# def resize(input_image, is_gray = True, width = None, height = None):
#     if is_gray:
#         image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

#     h, w = image.shape[:2]
#     if width == None and height != None:
#         width = int(height/h*w)
#     elif width != None and height == None:
#         height = int(width/w*width)

#     image = cv2.resize(image, (width, height))
    
#     return image