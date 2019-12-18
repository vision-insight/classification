
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

gpu_ids = [0]

# Convert model to be used on device
model = nn.DataParallel(model, device_ids = gpu_ids)
model = model.cuda(device  = 0)

criterion = nn.CrossEntropyLoss(weight = class_weights.cuda(), reduction = "sum")

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
scheduler = exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#####################  02  model training #####################################
num_epochs = 100
save_dir = "./output_models"
save_name = "age_7_resnet"


since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(1, num_epochs + 1):

    print("Epoch {}/{}".format(epoch, num_epochs))
    print("-" * 10)

    # Each epoch has a training and validation phase
    for phase in ["train", "valid"]:
        if phase == "train":
            model.train() # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode
        
        start_time = time.time()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.cuda(device = 0)
            labels = labels.cuda(device = 0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == "train":
            scheduler.step()

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
        
        duration = time.time() - start_time
        print("{} Loss: {:.4f} | Acc: {:.4f} | Time elapsed: {:.0f}m {:.0f}s".format(
               phase, epoch_loss, epoch_acc, duration // 60, duration % 60))

        # deep copy the model
        if phase == "valid" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts,
                       os.path.join(save_dir, "_".join([save_name, str(epoch), time_stamp()]) + '.pth'))
            
    print()

time_elapsed = time.time() - since
print("Training complete in {:.0f}m {:.0f}s".format(
       time_elapsed // 60, time_elapsed %60))
print("Best val Acc: {:4f}".format(best_acc))


