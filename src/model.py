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

def create_simple_model():

    simple_model = nn.Sequential(
        # layer 1
        nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(16),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        # layer 2
        nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        # layer 3
        nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.Flatten(),
        nn.Linear(64 * 28 * 28, 200)
    )

    return simple_model

def create_deep_wide():
    deep_wide_model = nn.Sequential(
        # layer 1
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),

        # layer 2
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),

        # layer 3
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),

        # layer 4
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),

        # layer 5
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),

        # layer 6
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ELU(),
        nn.AdaptiveMaxPool2d((3, 3)),

        # Linear layer
        nn.Flatten(),
        nn.Linear(512 * 3 * 3, 512),
        nn.ELU(),
        nn.Linear(512, 200),
    )

    return deep_wide_model

def create_vgg():

    vgg_model = nn.Sequential(

        # convolutional block 1
        nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # convolutional block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # convolutional block 3
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # convolutional block 4
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # convolutional block 5
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveMaxPool2d((3, 3)),

        # fully connected layers
        nn.Flatten(),
        nn.Linear(512*3*3, 512),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(512, 200)
    )

    return vgg_model
    




def create_own_model():
    # this is the new model

    new_model = nn.Sequential(
        # layer 1
        nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(16),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        # layer 2
        nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        # layer 3
        nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        # layer 4
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        nn.Dropout(0.3),
        # layer 5
        nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(256),
        nn.ELU(),
        nn.MaxPool2d(kernel_size = 2),
        # Bottleneck layer
        nn.Conv2d(256, 64, kernel_size = 1),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.Conv2d(64, 256, kernel_size = 1),
        nn.BatchNorm2d(256),
        nn.ELU(),
        # Layer 6
        nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
        nn.ELU(),
        nn.AdaptiveMaxPool2d((3, 3)),
        # Linear layers
        nn.Dropout(0.4),
        nn.Flatten(),
        nn.Linear(512 * 3 * 3, 512),
        nn.ELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 200),
    )

    return new_model


if __name__ == "__main__":
    m = create_efficientnet(num_classes=200)
    # Next row calculates the parameter number of our model
    #m = create_own_model()
    print("Model created:", m.__class__.__name__)
    print("Trainable parameters:", count_parameters(m))
    # Uncomment to see full architecture:
    # print(m)
    create_own_model()
