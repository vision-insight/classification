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

weights_file = "./output_models/age_7_resnet_43_20191218_120846.pth"
gpu_id = [0]

model = torch.load(weights_file)

############ 01 model define #################################
if isinstance(model, collections.OrderedDict):
    # define the network
    try:
        model_structure = models.resnet18(pretrained=False)
    except:
        print("fdaf")

    fc_inputs = model_structure.fc.in_features
    model_structure.fc = nn.Sequential(
                         nn.Linear(fc_inputs, 512),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(512, len(idx_and_class)), # Since 10 possible outputs
                         nn.LogSoftmax(dim=1) # For using NLLLoss()
                         )
    try:
        model_structure.load_state_dict(model)
        model = model_structure
        # Convert model to be used on device
        model = nn.DataParallel(model, device_ids = gpu_id )
        model = model.cuda(device = 0)
    except Exception as e:
        if "module" in str(e):
            model_structure = nn.DataParallel(model_structure)
            model_structure.load_state_dict(model)
            model = model_structure
            model = model.cuda(device = 0)

else:
    model = torch.load(weights_file)
    model = model.cuda(device = 0)


############ 03 testing ################################
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    label_list = []
    pred_label_list = []
    for images, labels in tqdm(dataloaders["test"]):
        label_list.extend(labels)

        images = images.cuda(device = 0)
        labels = labels.cuda(device = 0)

        outputs = model(images)
        log_probs, pred_labels = torch.max(outputs.data, 1)
        pred_label_list.extend(pred_labels.cpu())

    #print("overall acc: %.3f%%" % (overall_acc*100), overall_prec, ave_prec, recall)
    print('''[INFO] 
            overall  acc : %.2f
            mean     acc : %.2f
            overall prec : %.2f
            mean    prec : %.2f
            mean  recall : %.2f'''  %
            (overall_accuracy(label_list, pred_label_list), #overall_recall(label_list, pred_label_list),
            mean_accuracy(label_list, pred_label_list),
            overall_prec(label_list, pred_label_list),
            mean_prec(label_list, pred_label_list),
            mean_recall(label_list, pred_label_list))) 


