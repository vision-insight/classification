import os
import sys
import  numpy as np
from tqdm import tqdm
from data_prepare import *
from sklearn.metrics import accuracy_score
import collections
from torch import nn
import argparse
import cv2
from torchvision import transforms
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

base_path = "/media/D/lulei/classification"
sys.path.insert(0, base_path)
from tools.metrics import *
from tools.utils.model_zoo import MODELS


########### 00 load the model #####################################
#weights_file = "./output_models/vehicle_resnet50_21_20191219_001108.pth"
#weights_file = "./output_models/vehicle_resnet18_27_20191218_164439.pth"
#weights_file = "./output_models/vehicle_alexnet_99_20191220_021124.pth"
weights_file = "./gender/output_models/gender_res18_0.6271_31_best_20200313_192739.pth"

############ 01 model define #################################
model_struc = MODELS(class_num = len(class_to_index), with_wts = False).resnet18()

###########  02 load wts ############################
model = load_model_from_wts(model_struc, weights_file, gpu_id = [0])

############ 03 testing ################################
model.eval()

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='./images/')
args= parser.parse_args()

for path in os.listdir(args.input):
    #print(path)
    imgs=cv2.imread('./images/'+path)
    h,w = imgs.shape[0:2]
    title_x = int(w*0.7)
    title_y = int(h*0.1)
    #print(type(imgs))
    #img = torch.tensor(img)
    img = transforms.ToTensor()(imgs).unsqueeze(0)
    #print(img)
    line_thickness = round(0.002 * max(img.shape[0:2]) + 1)
    #line_thickness = 7
    font_thickness = max(line_thickness - 1, 1)
    #img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    gender_result =model(img)
    _, pred = torch.max(gender_result, 1)
    #soft = torch.nn.functional.softmax(out, 1)
    #sign = (pred.data.item())
    #print(sign)
    print(pred)

    #print(pred)
    if pred  == 0:
        cv2.putText(imgs, 'female', (title_x,title_y), cv2.FONT_HERSHEY_TRIPLEX,line_thickness, (0,0,255), font_thickness)
        cv2.imwrite('./gender/results/'+path+'.jpg', imgs)
        #cv2.imwrite(save_train_path+strTime1+'.jpg',show_frame)
    else:
        cv2.putText(imgs, 'male', (title_x,title_y), cv2.FONT_HERSHEY_TRIPLEX,line_thickness, (0,0,255), font_thickness)
        cv2.imwrite('./gender/results/'+path+'.jpg', imgs)

'''image = cv2.imread('./gender/1.jpg')
h,w = image.shape[0:2]
title_x = int(w*0.7)
title_y = int(h*0.1)
img = transforms.ToTensor()(image).unsqueeze(0)
line_thickness = round(0.002 * max(img.shape[0:2]) + 1)
font_thickness = max(line_thickness - 1, 1)

output = model(img)
_, pred = torch.max(output, 1)
print(pred)
if pred == 0:
    cv2.putText(image, 'female', (title_x,title_y), cv2.FONT_HERSHEY_TRIPLEX,line_thickness, (0,0,255), font_thickness)
    cv2.imwrite('1-1.jpg',image)
else:
    cv2.putText(image, 'male', (title_x,title_y), cv2.FONT_HERSHEY_TRIPLEX,line_thickness, (0,0,255), font_thickness)
    cv2.imwrite('1-1.jpg',image)
'''
