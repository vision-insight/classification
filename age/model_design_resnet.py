
## refered to https://github.com/spmallick/learnopencv/blob/master/Image-Classification-in-PyTorch/image_classification_using_transfer_learning_in_pytorch.ipynb
## https://github.com/spmallick/learnopencv/blob/master/Image-classification-pre-trained-models/Image_Classification_using_pre_trained_models.ipynb

import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os
from torch_utils import *
from data_prepare import *

from PIL import Image


# Load pretrained ResNet50 Model
resnet = models.resnet18(pretrained=True)

# Freeze model parameters (for transfer learning)
for param in resnet.parameters():
    param.requires_grad = False

# Change the first conv layer to adapt single channel input
# print(resnet50.modules)
w = resnet.conv1.weight
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet.conv1.weight = torch.nn.Parameter(w[:, :1, :, :])

# Change the final layer of ResNet50 Model for Transfer Learning
fc_inputs = resnet.fc.in_features

resnet.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(index_to_class)), # Since 10 possible outputs
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)

# Convert model to be used on device
resnet = resnet.to(device)

# Define Optimizer and Loss Function
loss_func = nn.NLLLoss()

optimizer = optim.Adam(resnet.parameters())








num_epochs = 30
trained_model, history = train_and_validate(resnet,
                                            train_data_loader,
                                            valid_data_loader,
                                            device,
                                            loss_func,
                                            optimizer,
                                            epochs=num_epochs,
                                            save_dir = "./output_models",
                                            save_name = "age")


history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.savefig(dataset+'_loss_curve.png')
plt.show()



plt.plot(history[:,2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig(dataset+'_accuracy_curve.png')
plt.show()


# Print the model to be trained
#summary(resnet50, input_size=(3, 224, 224), batch_size=bs, device='cuda')

