import os
import sys
import  numpy as np
from tqdm import tqdm
from data_prepare import *
from sklearn.metrics import accuracy_score
import collections
from torch import nn

base_path = "/data/lulei/classification"
sys.path.insert(0, base_path)
from tools.metrics import *
from tools.utils.model_zoo import MODELS


########### 00 load the model #####################################
#weights_file = "./output_models/vehicle_resnet50_21_20191219_001108.pth"
#weights_file = "./output_models/vehicle_resnet18_27_20191218_164439.pth"
#weights_file = "./output_models/vehicle_alexnet_99_20191220_021124.pth"
weights_file = "./output_models/vehicle_resnet50_0.9182_49_best_20191224_142352.pth"

############ 01 model define #################################
model_struc = MODELS(class_num = len(class_to_index), with_wts = False).resnet50()

###########  02 load wts ############################
model = load_model_from_wts(model_struc, weights_file, gpu_id = [0])

############ 03 testing ################################
model.eval() 
with torch.no_grad():

    label_list = []
    pred_label_list = []

    for images, labels in tqdm(dataloaders["test"]):
        label_list.extend(labels)

        images, labels = images.cuda(device = 0), labels.cuda(device = 0)

        outputs = model(images)
        log_probs, pred_labels = torch.max(outputs.data, 1)

        pred_label_list.extend(pred_labels.cpu())

    for images, labels in tqdm(dataloaders["valid"]):
        label_list.extend(labels)

        images, labels = images.cuda(device = 0), labels.cuda(device = 0)

        outputs = model(images)
        log_probs, pred_labels = torch.max(outputs.data, 1)

        pred_label_list.extend(pred_labels.cpu())

print('''
[INFO] 
    overall  acc : %.4f
    mean     acc : %.4f
    mean    prec : %.4f
    mean  recall : %.4f '''  %
    (overall_accuracy(label_list, pred_label_list), #overall_recall(label_list, pred_label_list),
    mean_accuracy(label_list, pred_label_list),
    mean_prec(label_list, pred_label_list),
    mean_recall(label_list, pred_label_list))) 

