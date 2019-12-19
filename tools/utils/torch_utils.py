import os
import copy
import time
import torch
import datetime

def time_stamp():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S') #.%f

def select_device(force_cpu=False):
    if force_cpu:
        cuda = False
        device = torch.device('cpu')
    else:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')

        if torch.cuda.device_count() > 1:
            device = torch.device('cuda' if cuda else 'cpu')
            print('Found %g GPUs' % torch.cuda.device_count())
            # print('Multi-GPU Issue: https://github.com/ultralytics/yolov3/issues/21')
            # torch.cuda.set_device(0)  # OPTIONAL: Set your GPU if multiple available
            # print('Using ', torch.cuda.device_count(), ' GPUs')

    print('Using %s %s\n' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))
    return device

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, save_dir = "./", prefix = "model"):
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
                           os.path.join(save_dir, "_".join([prefix, str(epoch), time_stamp()]) + '.pth'))
    
        print()
    
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
           time_elapsed // 60, time_elapsed %60))
    print("Best val Acc: {:4f}".format(best_acc))




def computeTestSetAccuracy(model, loss_criterion):
    '''
    Function to compute the accuracy on the test set
    Parameters
        :param model: Model to test
        :param loss_criterion: Loss Criterion to minimize
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_acc = 0.0
    test_loss = 0.0

    # Validation - No gradient tracking needed
    with torch.no_grad():

        # Set to evaluation mode
        model.eval()

        # Validation loop
        for j, (inputs, labels) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = loss_criterion(outputs, labels)

            # Compute the total loss for the batch and add it to valid_loss
            test_loss += loss.item() * inputs.size(0)

            # Calculate validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to valid_acc
            test_acc += acc.item() * inputs.size(0)

            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_data_size 
    avg_test_acc = test_acc/test_data_size

    print("Test accuracy : " + str(avg_test_acc))



def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    
    transform = image_transforms['test']

    test_image = Image.open(test_image_name)
    plt.imshow(test_image)
    
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)
        for i in range(3):
            print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])



# Test a particular model on a test image

# dataset = 'caltech_10'
# model = torch.load('caltech_10_model_8.pt')
# predict(model, 'pixabay-test-animals/triceratops-954293_640.jpg')
