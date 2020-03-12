import torch
import collections
import torch.nn as nn
from data_prepare import *
import torch.optim as optim
from torchvision import datasets, models, transforms

weights_file = "./output_models/age_resnet_77_20191213_142153.pth"



model = torch.load(weights_file)
print(type(model))
#class_num = model["fc.3.weight"].shape[0]
if isinstance(model, collections.OrderedDict):
    # define the network
    model_structure = models.resnet18(pretrained=False)

    fc_inputs = model_structure.fc.in_features
    model_structure.fc = nn.Sequential(
                         nn.Linear(fc_inputs, 512),
                         nn.ReLU(),
                         nn.Dropout(0.5),
                         nn.Linear(512, len(idx_and_class)), # Since 10 possible outputs
                         nn.LogSoftmax(dim=1) # For using NLLLoss()
                         )
    model_structure.load_state_dict(model) 
    model = model_structure
    # Convert model to be used on device
    model = nn.DataParallel(model, device_ids = [0,1] )
    model = model.cuda(device = 0)

elif isinstance(model, torch.nn.parallel.data_parallel.DataParallel):
    pass

#print(type(model))

loss_func = nn.NLLLoss(weight=class_weights.cuda(), reduction='sum')

optimizer = optim.Adam(model.parameters())




## model training #####################################
epochs = 100
start_num = 50
save_dir = "./output_models"
prefix = "age_resnet"
save_model = True
save_thre = 20
save_type = "full_model" #"full_model" # weights_only


start = time.time()
history = []
best_acc = 0.0

train_data_size = len(train_data_loader.dataset)
valid_data_size = len(valid_data_loader.dataset)

for epoch in range(start_num, epochs + 1):
    epoch_train_start = time.time()

    # Set to training mode
    model.train()

    # Loss and Accuracy within the epoch
    train_loss = 0.0
    train_acc = 0.0

    valid_loss = 0.0
    valid_acc = 0.0

    for i, (inputs, labels) in enumerate(train_data_loader):

        inputs = inputs.cuda(device = 0)
        labels = labels.cuda(device = 0)

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
            inputs = inputs.cuda(device = 0)
            labels = labels.cuda(device = 0)

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
            format(epoch, epochs, avg_train_loss,\
            avg_train_acc*100, epoch_valid_start - epoch_train_start ), end= " | ")
    print("Valid : Loss: {:.4f}, Acc: {:.2f}%, Time: {:.2f}s"\
            .format(avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_valid_start))

    # Save if the model has best accuracy till now
    if save_model and (epoch >= save_thre):
        file_path = os.path.join(save_dir, \
                                 "_".join([prefix, str(epoch), time_stamp()]) + '.pth')
        if save_type == "full_model":
            torch.save(model, file_path)
        elif save_type == "weights_only":
            torch.save(model.module.state_dict(), file_path)
        else:
            print("[INFO] invalid save type, unable to save the model")



