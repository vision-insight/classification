
## refered to https://github.com/spmallick/learnopencv/blob/master/Image-Classification-in-PyTorch/image_classification_using_transfer_learning_in_pytorch.ipynb
## https://github.com/spmallick/learnopencv/blob/master/Image-classification-pre-trained-models/Image_Classification_using_pre_trained_models.ipynb
import os
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from data_prepare import *
from PIL import Image
from torch.optim import lr_scheduler
import copy


###################### 00  model defination ###################################

# Load pretrained ResNet50 Model
model = models.resnet18(pretrained=True)

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
                    nn.Linear(fc_inputs, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, len(class_to_index)), # Since 10 possible outputs
                    nn.LogSoftmax(dim=1) # For using NLLLoss()
                        )

######################  01 training parameters ############################

gpu_ids = [0, 1]

# Convert model to be used on device
model = nn.DataParallel(model, device_ids = gpu_ids)
model = model.cuda(device  = 0)

#criterion = nn.CrossEntropyLoss(weight = class_weights.cuda(), reduction = "sum")
criterion = nn.NLLLoss(weight = class_weights.cuda(), reduction = "sum")

# Observe that all parameters are being optimized
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters())

# Decay LR by a factor of 0.1 every 20 epochs
scheduler = exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#####################  02  model training #####################################
num_epochs = 100
save_dir = "./output_models"
save_name = "age_4_resnet"

train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=num_epochs, save_dir = save_dir, prefix = save_name)

