import torch
from torch import nn
from torchvision import models

# EfficientNet model that uses the pretrained backbone and a custom classifier head
class BirdEfficientNet(nn.Module):

    def __init__(self, num_classes, model_name: str = "efficientnet_b0", pretrained = True, dropout = 0.3, hidden_dim = 512, activation = "relu"):
        super().__init__()

        if model_name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            base = models.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        self.backbone = base.features
        self.pool = base.avgpool
        self.flatten = nn.Flatten()

        # Original classifier input dim
        in_features = base.classifier[1].in_features

        # Choose activation
        if activation.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif activation.lower() == "gelu":
            act_layer = nn.GELU()
        elif activation.lower() == "silu":
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Custom head
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            act_layer,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   
        x = self.pool(x)       
        x = self.flatten(x)    
        logits = self.head(x)  
        return logits


def create_efficientnet(num_classes, model_name = "efficientnet_b0", pretrained = True, dropout = 0.3, hidden_dim = 512, activation = "relu"):
    return BirdEfficientNet(num_classes=num_classes, model_name=model_name, pretrained=pretrained, dropout=dropout, hidden_dim=hidden_dim, activation=activation)

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = create_efficientnet(num_classes=200)
    print("Model created:", m.__class__.__name__)
    print("Trainable parameters:", count_parameters(m))
    # Uncomment to see full architecture:
    # print(m)
