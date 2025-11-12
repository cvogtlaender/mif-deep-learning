import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetFinetune(nn.Module):

  def __init__(self, num_classes: int = 10, pretrained: bool = True, freeze_backbone: bool = False):
    super(ResNetFinetune, self).__init__()

    if pretrained:
      weights = ResNet18_Weights.IMAGENET1K_V1
      self.model = resnet18(weights=weights)
    else:
      self.model = resnet18(weights=None)

    if freeze_backbone:
      for param in self.model.parameters():
        param.requires_grad = False

    in_features = self.model.fc.in_features
    self.model.fc = nn.Linear(in_features, num_classes)

  def forward(self, x):
    return self.model(x)