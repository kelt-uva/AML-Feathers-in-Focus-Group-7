from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import BirdDataset, get_transforms
from model import create_efficientnet


def set_seed(seed = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes(train_csv_path):
    df = pd.read_csv(train_csv_path)
    return df["label"].nunique()


# Creates train and validation dataloaders
def create_dataloaders(train_csv_path, batch_size = 32, val_frac = 0.1, image_size = 224):
    full_dataset = BirdDataset(csv_path=train_csv_path, transform=get_transforms(True, image_size=image_size), has_labels=True)

    num_samples = len(full_dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(num_samples * (1 - val_frac))
    train_idx, val_idx = indices[:split], indices[split:]

    train_subset = Subset(full_dataset, train_idx)

    # Validation dataset
    val_dataset = BirdDataset(csv_path=train_csv_path, transform=get_transforms(False, image_size=image_size), has_labels=True)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)             
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    set_seed(42)

    project_root = Path(__file__).resolve().parents[0]  
    train_csv_path = project_root / "train_images.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)


    # Hyperparameters
    batch_size = 32
    num_epochs = 10
    image_size = 224
    learning_rate = 1e-4
    model_name = "efficientnet_b0"

    # Infer number of classes from train CSV
    num_classes = get_num_classes(train_csv_path)
    print("Inferred num_classes:", num_classes)

    # Dataloaders
    train_loader, val_loader = create_dataloaders(train_csv_path=train_csv_path, batch_size=batch_size, val_frac=0.1, image_size=image_size)

    # Model, loss, optimizer
    model = create_efficientnet(num_classes=num_classes, model_name=model_name, pretrained=True, dropout=0.2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    best_val_acc = 0.0
    best_model_path = project_root / "best_efficientnet_b0.pth"

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train  - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Val    - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} (val_acc={best_val_acc:.4f})")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
