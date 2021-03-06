import os
import sys
import  numpy as np
from tqdm import tqdm
from data_prepare_old import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import collections
from torch import nn

base_path = "/data/lulei/classification"
sys.path.insert(0, base_path)
from tools.metrics import *
from tools.utils.model_zoo import MODELS


########### 00 load the model #####################################
weight_files = [i for i in pathlib.Path("./output_models").rglob("*.pth")]

print(f"weight files : ")
for index, weight_file in enumerate(weight_files):
    print(f"[{index}] {weight_file.name}")
weight_file = weight_files[int(input("weight index : "))]

network = os.path.basename(weight_file).split("_")[1]

model_struc = eval(f"MODELS(class_num = n_classes, with_wts = False).{network}()")


###########  02 load wts ############################
model = load_model_from_wts(model_struc, weight_file, gpu_id = [0,1])

############ 03 testing ################################
y_true, y_pred = [], []
model.eval() 
with torch.no_grad():

    #for images, labels in tqdm(dataloaders["valid"]):
    #    y_true.extend(labels)
#
    #    outputs = model(images.cuda(device = 0))
    #    log_probs, pred_labels = torch.max(outputs.data, 1)
#
    #    y_pred.extend(pred_labels.cpu())
        
    for images, labels in tqdm(dataloaders["test"]):
        y_true.extend(labels)
        
        outputs = model(images.cuda(device = 0))
        log_probs, pred_labels = torch.max(outputs.data, 1)
        
        y_pred.extend(pred_labels.cpu())
        
        


#print('''
#[INFO] 
#    overall  acc : %.4f
#    mean     acc : %.4f
#    mean    prec : %.4f
#    mean  recall : %.4f '''  %
#    (overall_accuracy(y_true, y_pred), #overall_recall(label_list, y_pred),
#     mean_accuracy(y_true, y_pred),
#    mean_accuracy(y_true, y_pred),
#    mean_prec(y_true, y_pred),
#    mean_recall(y_true, y_pred))) 

print(f"Performance : ")
metrics = [mean_accuracy]
prefix  = ["mean acc"]
for describe, metric in zip(prefix, metrics):
    print(f"  {describe.ljust(15)} : {metric(y_true, y_pred):.4f}")
#result = classification_report(y_true, y_pred) #, target_names=target_names)
#print(result[:10])