import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTFinetune(nn.Module):

    def __init__(self, num_classes: int = 10, pretrained: bool = True, freeze_backbone: bool = False):
        super(ViTFinetune, self).__init__()

        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.model = vit_b_16(weights=weights)
        else:
            self.model = vit_b_16(weights=None)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
