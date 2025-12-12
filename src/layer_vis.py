from model import create_own_model
from dataset import BirdDataset, get_transforms
from pathlib import Path
import torch.nn as nn
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def set_seed(seed = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fetch_data(test_csv_path, image_size = 224):
    test_dataset = BirdDataset(csv_path=test_csv_path, transform=get_transforms(False, image_size=image_size, example=True), has_labels=False)
    return test_dataset

def get_model_blocks(model):
    blocks = []
    current_block = []

    for layer in model:
        current_block.append(layer)
        if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveMaxPool2d):
            blocks.append(nn.Sequential(*current_block))
            current_block = []

    # last block
    if len(current_block) > 0:
        return blocks

def visualize_feature_maps(tensor, max_channels=16, title="", save_path=None):
    tensor = tensor[0]
    C, H, W = tensor.shape

    C = min(C, max_channels)
    
    fig, axes = plt.subplots(1, C, figsize=(2*C, 2))

    for i in range(C):
        fmap = tensor[i].detach().cpu()
        
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)

        axes[i].imshow(fmap, cmap="Grays")
        axes[i].axis("off")
        axes[i].set_title(f"Channel {i+1}")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

def main():

    set_seed(42)

    project_root = Path(__file__).resolve().parents[0]  
    test_csv_path = project_root / "test_images_path.csv"

    save_path = project_root / "./example/layer_vis{}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    save_path = str(save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    print("Using device:", device)

    model = create_own_model()
    model.to(device)
    model.load_state_dict(torch.load(project_root / "best_custom_model.pth"))
    model.eval()

    print("Model loaded and set to evaluation mode.")

    test_dataset = fetch_data(test_csv_path=test_csv_path, image_size=224)

    test_dataset_lentgh = test_dataset.__len__()

    random_image = test_dataset[np.random.randint(0, test_dataset_lentgh)]
    image_tensor = random_image['image'].unsqueeze(0).to(device)

    blocks = get_model_blocks(model)

    plt.imshow(np.transpose(random_image['image'].cpu().numpy(), (1, 2, 0)))
    plt.axis("off")
    plt.title("Input Image")
    plt.savefig(save_path.format("input"), dpi=300)

    x = image_tensor
    for i, block in enumerate(blocks):
        x = block(x)
        print(f"Block {i+1}: shape {x.shape}")

        visualize_feature_maps(
            x, 
            max_channels=6, 
            title=f"Block {i+1} Output Feature Maps",
            save_path=save_path.format(i+1)
        )


if __name__ == "__main__":
    main()