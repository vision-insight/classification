from data_prepare import *
from sklearn import metrics

# torch.set_printoptions(precision=5, threshold=None, edgeitems=None, linewidth=None, profile=None)



weights_file = "./model_30_20191120_152132.pt"
model = torch.load(weights_file)
model = model.to(device)

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    label_list = []
    pred_label_list = []
    for images, labels in test_data_loader:
        label_list.extend(labels)

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        log_probs, pred_labels = torch.max(outputs.data, 1)
        pred_label_list.extend(pred_labels)

    overall_acc = metrics.accuracy_score(label_list, pred_label_list)
    recall = metrics.recall_score(label_list, pred_label_list, average='macro') #None
    print(overall_acc, recall)
    



