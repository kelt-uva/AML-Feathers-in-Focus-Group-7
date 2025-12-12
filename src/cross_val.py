from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import BirdDataset, get_transforms
from model import create_efficientnet, create_own_model, create_simple_model, create_deep_wide, create_vgg
import time

model = create_own_model()

def set_seed(seed=777):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
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
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1, all_labels, all_preds

def plot_cv_results(results_df, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Bar plot comparing folds
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df))
    width = 0.25
    
    ax.bar(x - width, results_df['val_acc'], width, label='Accuracy', alpha=0.8)
    ax.bar(x, results_df['val_f1'], width, label='F1 Score', alpha=0.8)
    ax.bar(x + width, results_df['train_acc'], width, label='Train Acc', alpha=0.8)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Results by Fold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i+1}' for i in range(len(results_df))])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'cv_folds_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved to {save_dir}")

def main(n_splits=5, num_epochs=30, batch_size=32, learning_rate=1e-4, weight_decay=5e-4, model=model):
    set_seed(777)
    
    project_root = Path(__file__).resolve().parent
    train_csv_path = project_root / "train_images.csv"
    results_dir = project_root / "results"
    vis_dir = project_root / "vis"
    results_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Batch size: {batch_size}, LR: {learning_rate}, Weight decay: {weight_decay}\n")
    df = pd.read_csv(train_csv_path)
    labels = (df["label"].astype(int) - 1).values  
    num_classes = df["label"].nunique()
    print(f"Dataset size: {len(df)}, Number of classes: {num_classes}\n")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
    
    # Results for each fold
    fold_results = []
    all_fold_cm = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Datasets for this fold
        train_dataset = BirdDataset(
            csv_path=train_csv_path,
            transform=get_transforms(True, image_size=224),
            has_labels=True
        )
        val_dataset = BirdDataset(
            csv_path=train_csv_path,
            transform=get_transforms(False, image_size=224),
            has_labels=True
        )
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_train_acc = 0.0
        best_epoch = 0
        
        fold_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_f1, val_labels, val_preds = validate_one_epoch(
                model, val_loader, criterion, device
            )
            
            # Store history
            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['train_f1'].append(train_f1)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)
            fold_history['val_f1'].append(val_f1)
            
            # Best metrics
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_train_acc = train_acc
                best_epoch = epoch
                best_val_labels = val_labels
                best_val_preds = val_preds
            
            if epoch % 5 == 0 or epoch == num_epochs:
                print(f"Epoch {epoch:2d}/{num_epochs} | "
                      f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f} | "
                      f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f}")
        
        print(f"\nFold {fold} Best Results (epoch {best_epoch}):")
        print(f"  Val Accuracy: {best_val_acc:.4f}")
        print(f"  Val F1 Score: {best_val_f1:.4f}")
        print(f"  Train Accuracy: {best_train_acc:.4f}")
        
        # Store fold results
        fold_results.append({
            'fold': fold,
            'train_acc': best_train_acc,
            'val_acc': best_val_acc,
            'val_f1': best_val_f1,
            'val_loss': min(fold_history['val_loss']),
            'best_epoch': best_epoch
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, num_epochs + 1)
        
        # F1 plot
        axes[0].plot(epochs_range, fold_history['val_f1'], label='Val F1', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('F1 score')
        axes[0].set_title(f'Fold {fold} - F1')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs_range, fold_history['train_acc'], label='Train Acc', alpha=0.8)
        axes[1].plot(epochs_range, fold_history['val_acc'], label='Val Acc', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Fold {fold} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / f'fold_{fold}_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Results dataframe
    results_df = pd.DataFrame(fold_results)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    results_df.to_csv(
        results_dir / f'cross_validation_results_{timestamp}.csv',
        index=False
    )
    print(f"\nResults saved to: {results_dir / f'cross_validation_results_{timestamp}.csv'}")
    
    plot_cv_results(results_df, vis_dir)
    
    print("\nCompleted")
    return results_df

if __name__ == "__main__":
    results = main(
        n_splits=5,          
        num_epochs=30,       
        batch_size=32,       
        learning_rate=1e-4,  
        weight_decay=5e-4    
    )