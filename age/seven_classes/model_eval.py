import os
import sys
import  numpy as np
from tqdm import tqdm
from data_prepare import *
from sklearn import metrics
sys.path.insert(0, "../")
sys.path.insert(0, "../../")
from tools.metrics import *

#torch.set_printoptions(precision=3, threshold=8, edgeitems=None, linewidth=None, profile=None)


weights_file = "./output_models/age_resnet_30_20191128_050216.pt"
model = torch.load(weights_file)
model = model.to(device)

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    label_list = []
    pred_label_list = []
    for images, labels in tqdm(valid_data_loader):
        label_list.extend(labels)

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        log_probs, pred_labels = torch.max(outputs.data, 1)
        pred_label_list.extend(pred_labels.cpu())

    overall_acc = metrics.accuracy_score(label_list, pred_label_list)
    overall_recall = metrics.recall_score(label_list, pred_label_list, average='micro') #None
    over_prec = metrics.precision_score(label_list, pred_label_list, average = "micro")
    ave_prec = metrics.precision_score(label_list, pred_label_list, average = "macro")
    ave_recall = metrics.recall_score(label_list, pred_label_list, average = "macro")

    #print("overall acc: %.3f%%" % (overall_acc*100), overall_prec, ave_prec, recall)
    print(overall_acc, over_prec, ave_prec, ave_recall)
    print(overall_accuracy(label_list, pred_label_list),
            #overall_recall(label_list, pred_label_list),
            overall_prec(label_list, pred_label_list),
            mean_prec(label_list, pred_label_list),
            mean_recall(label_list, pred_label_list)) 


