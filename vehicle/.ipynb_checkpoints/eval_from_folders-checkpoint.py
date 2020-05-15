import os
import sys
import  numpy as np
from tqdm import tqdm
from data_prepare_old import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import collections
from torch import nn
from pathlib import Path

base_path = "/data/lulei/classification"
sys.path.insert(0, base_path)
from tools.metrics import *
from tools.utils.utils import *
from tools.utils.model_zoo import MODELS

test_dir = "/data/lulei/data/vehicle/frontal_103/split/test"
valid_dir = "/data/lulei/data/vehicle/frontal_103/split/valid"

image_paths = [i for i in Path(test_dir).rglob("*.jpg") ] + [i for i in Path(valid_dir).rglob("*.jpg")]
print(f"[INFO] image num : {len(image_paths)}")

class_num = {}
for i in image_paths:
    label = i.parts[-2]
    if label in class_num:
        class_num[label] += 1
    else:
        class_num.update({label: 1})

class_num = sorted(class_num.items(), key = lambda kv:(kv[1], int(kv[0])), reverse = True)

top_5_classes = [i[0] for i in class_num[:5]]
bottom_5_classes = [i[0] for i in class_num[-5:]]


top_5_images = [ i for i in image_paths if i.parts[-2] in top_5_classes]
bottom_5_images = [ i for i in image_paths if i.parts[-2] in bottom_5_classes]
print(f"[INFO] top 5 classes, image num : {len(top_5_images)}")
print(f"[INFO] bot 5 classes, image num : {len(bottom_5_images)}")
      

trans = transforms.Compose([
                transforms.Resize((227, 227), interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor()])

def acquire_task(image_list, divide = 80):
    sub_groups = list_split(image_list, divide)
    batch_list = []
    for group in sub_groups:
        temp_batch = []
        temp_labels = []
        for path in group:
            label = path.parts[-2]
            image = trans(Image.open(path)).unsqueeze(0)
            temp_batch.append(image)
            temp_labels.append(label)
        batch_list.append([temp_batch, temp_labels])
    return batch_list



weight_files = [i for i in pathlib.Path("./output_models").rglob("*.pth")]
print(f"weight files : ")
for index, weight_file in enumerate(weight_files):
    print(f"[{index}] {weight_file.name}")
weight_file = weight_files[int(input("weight index : "))]
network = os.path.basename(weight_file).split("_")[1]

networks = ["alexnet", "densenet121", "resnet18", "resnet34", "resnet50", "vgg16", "vgg19"]

model_struc = eval(f"MODELS(class_num = 1759, with_wts = False).{network}()")

###########  02 load wts ############################
model = load_model_from_wts(model_struc, weight_file, gpu_id = [0,1])



y_true, y_pred = [], []
batch_list = acquire_task(top_5_images, divide = 80)
for batch, labels in batch_list:
    y_true.extend(labels)
    batch_input = torch.cat(batch,0)#.to(config.device) 
    with torch.no_grad():
        batch_out = model(batch_input.cuda(device = 0))
        probs, pred_labels = torch.max(batch_out.data,1)
        y_pred.extend([index_to_class[i] for i in pred_labels.cpu()])
        print(y_pred)
    
print('''
[INFO] 
    overall  acc : %.4f
    mean     acc : %.4f
    mean    prec : %.4f
    mean  recall : %.4f '''  %
    (overall_accuracy(y_true, y_pred), #overall_recall(label_list, y_pred),
    mean_accuracy(y_true, y_pred),
    mean_prec(y_true, y_pred),
    mean_recall(y_true, y_pred))) 

#result = classification_report(y_true, y_pred) #, target_names=target_names)
print(result[:10])
        