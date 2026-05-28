import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ReIDResNet(nn.Module):
    def __init__(self, num_classes=491):
        super(ReIDResNet, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False) 
        self.classifier = nn.Linear(2048, num_classes, bias=False)
        self.bottleneck.apply(self.weights_init_kaiming)
        self.classifier.apply(self.weights_init_classifier)

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features).view(features.size(0), -1)
        feat_bn = self.bottleneck(features)
        
        if self.training:
            cls_score = self.classifier(feat_bn)
            return cls_score, features, feat_bn
        else:
            return feat_bn
