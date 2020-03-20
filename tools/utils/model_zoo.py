from torchvision import models
import torch.nn as nn




class MODELS:
    def __init__(self, class_num, with_wts = True):
        self.class_num = class_num
        self.with_wts = with_wts

    def alexnet(self):
        model = models.alexnet(pretrained = self.with_wts)
        model.classifier[6] = nn.Linear(in_features=model.classifier[4].out_features, \
                                            out_features= self.class_num, bias=True)
        return model

    def densenet121(self):
        model = models.densenet121(pretrained = self.with_wts)
        model.classifier = nn.Linear(in_features=1024, out_features=self.class_num, bias=True)
        return model

    def resnet18(self):
        model = models.resnet18(pretrained = self.with_wts)
        model.fc = nn.Linear(in_features=model.fc.in_features, \
                                    out_features= self.class_num, bias=True)
        return model 

    def resnet34(self):
        model = models.resnet34(pretrained = self.with_wts)
        model.fc = nn.Linear(in_features=model.fc.in_features, \
                                    out_features= self.class_num, bias=True)
        return model

    def resnet50(self):
        model = models.resnet50(pretrained = self.with_wts)
        model.fc = nn.Linear(in_features=model.fc.in_features, \
                                    out_features= self.class_num, bias=True)
        return model

    def vgg16(self):
        model = models.vgg16(pretrained = self.with_wts)
        model.classifier[6] = nn.Linear(in_features=4096, \
                                    out_features= self.class_num, bias=True)
        return model

    def vgg19(self):
        model = models.vgg19(pretrained = self.with_wts)
        model.classifier[6] = nn.Linear(in_features=4096, \
                                    out_features= self.class_num, bias=True)
        return model


