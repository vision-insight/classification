
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
from utils.torch_utils import *
from data_prepare import *

from PIL import Image


# Load pretrained ResNet50 Model
model = models.densenet121(pretrained=True)

# uncomment to use transfer learning
# # Freeze model parameters (for transfer learning)
# for param in model.parameters():
#     param.requires_grad = False

# Change the first conv layer to adapt single channel input
# w = model.conv1.weight
#model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#model.conv1.weight = torch.nn.Parameter(w[:, :1, :, :])

# Change the final layer of ResNet50 Model for Transfer Learning
num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, 2)

model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(class_to_index)), # Since 10 possible outputs
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)

# Convert model to be used on device
model = model.to(device)
model = nn.DataParallel(model)
# Define Optimizer and Loss Function
loss_func = nn.NLLLoss(weight=class_weights.cuda(), reduction='sum')

optimizer = optim.Adam(model.parameters())



## model training #####################################
epochs = 200
save_dir = "./output_models"
save_name = "age_dense"

start = time.time()
history = []
best_acc = 0.0

train_data_size = len(train_data_loader.dataset)
valid_data_size = len(valid_data_loader.dataset)

for epoch in range(1, epochs + 1):
    epoch_train_start = time.time()

    # Set to training mode
    model.train()

    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0

    valid_loss = 0.0
    valid_acc = 0.0

    for i, (inputs, labels) in enumerate(train_data_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Clean existing gradients
        optimizer.zero_grad()

        # Forward pass - compute outputs on input data using the model
        outputs = model(inputs)

        # Compute loss
        loss = loss_func(outputs, labels,)

        # Backpropagate the gradients
        loss.backward()
            
        # Update the parameters
        optimizer.step()

        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)

        # Compute the accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))

            # correct = (predictions == labels).sum().float()
            # print(correct_counts, correct, type(correct_counts), type(correct))

            # Convert correct_counts to float and then compute the mean
        acc = torch.mean(correct_counts.type(torch.FloatTensor))

        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += acc.item() * inputs.size(0)


    epoch_valid_start = time.time()
    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(valid_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_func(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            valid_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)

            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            valid_acc += acc.item() * inputs.size(0)
    # Find average training loss and training accuracy
    avg_train_loss = train_loss/train_data_size
    avg_train_acc = train_acc/train_data_size

    # Find average training loss and training accuracy
    avg_valid_loss = valid_loss/valid_data_size
    avg_valid_acc = valid_acc/valid_data_size

    history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

    epoch_end = time.time()

    print("Epoch : {:d}/{:d}, Train : Loss: {:.4f}, Acc: {:.2f}%, Time: {:.2f}s".\
            format(epoch, epochs, avg_train_loss, avg_train_acc*100, epoch_valid_start - epoch_train_start ), end= " | ")
    print("Valid : Loss: {:.4f}, Acc: {:.2f}%, Time: {:.2f}s"\
            .format(avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_valid_start))

    # Save if the model has best accuracy till now
    #torch.save(model, os.path.join(save_dir, save_name + "_" + str(epoch) + "_" + time_stamp() + '.pt'))







#trained_model, history = train_and_validate(resnet,
#                                            train_data_loader,
#                                            valid_data_loader,
#                                            device,
#                                            loss_func,
#                                            optimizer,
#                                            epochs=num_epochs,
#                                            save_dir = "./output_models",
#                                            save_name = "age")


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

