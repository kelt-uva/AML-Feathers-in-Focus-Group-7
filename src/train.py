from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from dataset import BirdDataset, get_transforms
from model import create_efficientnet, create_own_model
import time
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score


def set_seed(seed = 777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_classes(train_csv_path):
    df = pd.read_csv(train_csv_path)
    return df["label"].nunique()

# Creates train and validation dataloaders
def create_dataloaders(train_csv_path, batch_size = 256, val_frac = 0.1, image_size = 224):
    full_dataset = BirdDataset(csv_path=train_csv_path, transform=get_transforms(True, image_size=image_size), has_labels=True)

    num_samples = len(full_dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split = int(num_samples * (1 - val_frac))
    train_idx, val_idx = indices[:split], indices[split:]

    train_subset = Subset(full_dataset, train_idx)

    # Weighted sampling
    train_val_info = pd.read_csv(train_csv_path)
    train_info = train_val_info.iloc[train_idx]
    train_labels = train_info["label"]
    num_train_samples = len(train_labels)
    train_label_counts = train_labels.value_counts()
    sample_weights = [1/train_label_counts[sample] for sample in train_info.label.values]
    sampler = WeightedRandomSampler(weights = sample_weights, num_samples = num_train_samples, replacement = True)



    # Validation dataset
    val_dataset = BirdDataset(csv_path=train_csv_path, transform=get_transforms(False, image_size=image_size), has_labels=True)
    val_subset = Subset(val_dataset, val_idx)

    # Remove sampler = sampler if you want to use fixed learning rate
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, sampler = sampler, num_workers=1, pin_memory=True, persistent_workers = True)

    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, persistent_workers = True)

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Collecting labels and predictions
    all_labels = []
    all_preds = []

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)             
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average = "macro")
    return epoch_loss, epoch_acc, epoch_f1


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Collecting labels and predictions
    all_labels = []
    all_preds = []

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

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average = "macro")
    return epoch_loss, epoch_acc, epoch_f1

def visualize_loss(epoch, train_loss, val_loss, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.plot(epoch, train_loss, color = "blue", label = "Training loss")
    ax.plot(epoch, val_loss, color = "orange", label = "Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss across epochs")
    plt.legend()
    plt.grid(True)
    fig.savefig(save_path)
    print("loss figure saved")
    plt.close(fig)

def visualize_accuracy(epoch, train_acc, val_acc, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.plot(epoch, train_acc, color = "blue", label = "Training accuracy")
    ax.plot(epoch, val_acc, color = "orange", label = "Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and validation accuracy across epochs")
    plt.legend()
    plt.grid(True)
    fig.savefig(save_path)
    print("accuracy figure saved")
    plt.close(fig)

def main(batch_size = 32, num_epochs = 50, learning_rate = 2e-4, weight_decay = 1e-2):
    set_seed(777)

    project_root = Path(__file__).resolve().parents[0]  
    train_csv_path = project_root / "train_images.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    print("Using device:", device)


    # Hyperparameters
    image_size = 224
    model_name = "efficientnet_b0"
    val_frac = 0.1

    # Infer number of classes from train CSV
    num_classes = get_num_classes(train_csv_path)
    print("Inferred num_classes:", num_classes)

    # Dataloaders
    train_loader, val_loader = create_dataloaders(train_csv_path=train_csv_path, batch_size=batch_size, val_frac = val_frac, image_size=image_size)

    # Model, loss, optimizer
    # model = create_efficientnet(num_classes=num_classes, model_name=model_name, pretrained=True, dropout=0.2).to(device)
    model = create_own_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    df_csv = pd.read_csv(train_csv_path)
    data_size = len(df_csv)
    train_size = data_size * (1 - val_frac)
    step_per_epoch = train_size/batch_size
    num_training_steps = num_epochs * step_per_epoch
    num_warmup_steps = num_training_steps * 0.1
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps)

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_path = project_root / "best_custom_model.pth"

    current_epoch = []
    train_acc_epochs = []
    val_acc_epochs = []
    train_f1_epochs = []

    train_loss_epoch = []
    val_loss_epoch = []
    val_f1_epochs = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # To disable the scheduler, set it to None
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler = scheduler)
        print(f"Train  - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

        val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Val    - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} (val_acc={best_val_acc:.4f}) (val_f1 = {best_val_f1})")
        
        # Tracking best f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
    
        # Saving losses, accuracies, and f1s into variables
        current_epoch.append(epoch)
        train_acc_epochs.append(train_acc)
        val_acc_epochs.append(val_acc)

        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)

        train_f1_epochs.append(train_f1)
        val_f1_epochs.append(val_f1)

    # Create timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path_loss = f"./vis/loss{timestamp}.png"
    save_path_accuracy = f"./vis/acc{timestamp}.png"
    # visualization of training and validation loss and accuracy
    visualize_loss(current_epoch, train_loss_epoch, val_loss_epoch, save_path_loss)
    visualize_accuracy(current_epoch, train_acc_epochs, val_acc_epochs, save_path_accuracy)

    # Creating a dataframe of the results and saving them to csv file
    all_metrics = np.array([current_epoch, train_loss_epoch, train_acc_epochs, val_loss_epoch , val_acc_epochs])
    all_metrics_df = pd.DataFrame(all_metrics, index = ['Epoch', 'Train_loss', 'Train_acc', 'Val_loss', 'Val_acc'])

    # Saving the metrics
    os.makedirs("./results", exist_ok = True)
    all_metrics_df.to_csv(f"./results/All_metrics_{timestamp}.csv")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f} Best validation f1: {best_val_f1}")

    # for crossval file
    return best_val_acc, best_val_f1

if __name__ == "__main__":
    main()
