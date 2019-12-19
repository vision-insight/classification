import os
import sys
import  numpy as np
from tqdm import tqdm
from data_prepare import *
from sklearn import metrics
import collections
from torch import nn

base_path = "/data/lulei/classification"
sys.path.insert(0, base_path)
from tools.metrics import *


########### 00 load the model #####################################
#weights_file = "./output_models/vehicle_resnet50_21_20191219_001108.pth"
weights_file = "./output_models/vehicle_resnet18_27_20191218_164439.pth"

def load_model_from_wts(model_struc, weights, gpu_id = [0]):
    wts = torch.load(weights)
    if isinstance(wts, collections.OrderedDict):
        try:
            model_struc.load_state_dict(wts)
            # Convert model to be used on device
            model = nn.DataParallel(model_struc, device_ids = gpu_id )
            model = model.cuda(device = 0)
        except Exception as e:
            if "module" in str(e):
                model = nn.DataParallel(model_struc)
                model.load_state_dict(wts)
                model = model.cuda(device = 0)
    else:
        raise Exception("Invalid weight file", weights) 

    return model


############ 01 model define #################################
model_struc = models.resnet18(pretrained=False)
model_struc.fc = nn.Linear(in_features=model_struc.fc.in_features,\
                                         out_features=len(class_to_index), bias=True)
model = load_model_from_wts(model_struc, weights_file, gpu_id = [0])

############ 03 testing ################################
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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


