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


image_dir = "./input_images/"
output_dir = "./output_images/"


data = ImageFolder(root = image_dir,
                   transform=image_trans["valid"])

dataloader = DataLoader(data,
                        batch_size = batch_size,
                        shuffle = True,
                        num_workers = cpu_count()//4*3)

########### 00 load the model #####################################
weights_file = "./output_models/gender_res34_0.9673_24_best_20200324_200609.pt"

############ 01 model define #################################
model_struc = MODELS(class_num = len(class_to_index), with_wts = False).resnet34()

###########  02 load wts ############################
model = load_model_from_wts(model_struc, weights_file, gpu_id = [0])

############ 03 testing ################################



model.eval() 
with torch.no_grad():

    pred_label_list = []

    for image_path  in tqdm(pathlib.Path(image_dir).rglob("*.jpg")):
        image = Image.open(image_path)
        image_torch = image_trans["valid"](image).unsqueeze(0).cuda(device = 0)
        output = model(image_torch)
        log_prob, pred_label = torch.max(output.data, 1)
        label = index_to_class[int(pred_label.cpu())]
        
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        cv2.putText(image, f"{label}", (20,20),\
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, f"{label}", (20,20), \
                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        
        out_dir = os.path.join(output_dir, label)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        out_image_path = os.path.join(out_dir,  os.path.basename(str(image_path)))
        cv2.imwrite(out_image_path, image)
