from  tqdm import tqdm
from data_prepare import *
from sklearn import metrics

#torch.set_printoptions(precision=3, threshold=8, edgeitems=None, linewidth=None, profile=None)


weights_file = "./output_models/age_resnet_169_20191127_004826.pt"
model = torch.load(weights_file)
model = model.to(device)

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    label_list = []
    pred_label_list = []
    for images, labels in tqdm(test_data_loader):
        label_list.extend(labels)

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        log_probs, pred_labels = torch.max(outputs.data, 1)
        pred_label_list.extend(pred_labels.cpu())

    overall_acc = metrics.accuracy_score(label_list, pred_label_list)
    recall = metrics.recall_score(label_list, pred_label_list, average='macro') #None
    print("overall acc: %.3f%%" % (overall_acc*100))
    



