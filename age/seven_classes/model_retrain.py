import torch


weights_file = "/data/lulei/detect_server/weights/best_has_person_resnet18_gray.pkl"



model = torch.load(weights_file)
print(type)
model = model.cuda(device = 0)
daf


self.model = models.resnet18(pretrained=False)
num_ftrs = self.model.fc.in_features
self.model.fc = nn.Linear(num_ftrs, 2)


self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

self.model.load_state_dict(torch.load(weights))
self.model = self.model.to(self.device)

self.model.eval()
self.trans = transforms.Compose([
                        transforms.Grayscale(num_output_channels=3),
                        transforms.Resize((360, 640)),
                        transforms.ToTensor()
                    ])




