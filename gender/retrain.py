import os
import sys
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
#from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from data_prepare import *
from PIL import Image
from torch.optim import lr_scheduler
import copy

base_path = "/media/D/lulei/classification/"
sys.path.insert(0, base_path) 
from tools.utils.model_zoo import MODELS

weights_file = "./output_models/gender_res18_0.8953_9_best_20200318_125440.pt"

model = MODELS(with_wts = False, class_num = 2).resnet18()

model = load_model_from_wts(model, weights_file, gpu_id = [0])

model = model.cuda(device  = 0)

criterion = nn.CrossEntropyLoss(weight = class_weights.cuda(), reduction = "sum")

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

#####################  02  model training #####################################
num_epochs = 30
save_dir = "./output_models"
save_name = "gender_retrain_res18"


train_model(model, dataloaders, criterion, optimizer, \
                   scheduler, num_epochs, save_dir = save_dir, prefix = save_name)
