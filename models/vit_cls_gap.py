import timm
from torch import nn

class ViT_CLS_GAP(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        self.embed_dim = self.backbone.embed_dim

        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        tokens = self.backbone.forward_features(x)

        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        gap = patch_tokens.mean(dim=1)
        combined = cls_token + gap

        out = self.classifier(combined)
        return out
