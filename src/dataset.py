from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os



# Applies general transormations to images
def get_transforms(train, image_size, example=False):
    if not example:
        base = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    else:
        base = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]

    if train:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.RandomAffine(degrees = 0, translate = (0.05, 0.05)),
            transforms.RandomRotation(degrees = 0.45),
        ]
        return transforms.Compose(aug + base + [transforms.RandomErasing()])

    return transforms.Compose(base)



class BirdDataset(Dataset):
    def __init__(self, csv_path, transform: None, has_labels = True, return_attributes = False, class_names_path = None, attributes_path = None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.has_labels = has_labels
        self.return_attributes = return_attributes

        src_dir = Path(__file__).resolve().parent              
        repo_root = src_dir.parent                            
        project_root = repo_root.parent                        
        cwd_root = Path.cwd()                                 

        self.roots = [
            src_dir / "train_images", 
            src_dir / "test_images",    
            src_dir,
            repo_root,
            project_root,
            cwd_root,
        ]

        if return_attributes:
            class_map_obj = np.load(class_names_path, allow_pickle=True).item()
            self.class_names = class_map_obj
            self.attributes = np.load(attributes_path, allow_pickle=True)
        else:
            self.attributes = None

    def __len__(self) -> int:
        return len(self.df)

    # Tries multiple plausible roots to resolve actual file path (had some trouble with train/test_images' paths)
    def _resolve_image_path(self, image_path_str):
        
        p = Path(image_path_str)

        if p.is_absolute():
            rel = p.relative_to("/")    
        else:
            rel = p

        tried = []
        for root in self.roots:
            candidate = root / rel
            tried.append(str(candidate))
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            f"Could not find image for '{image_path_str}'. "
            f"Tried:\n" + "\n".join(tried)
        )

    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rel_path = row["image_path"]
        img_path = self._resolve_image_path(rel_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        sample: Dict[str, Any] = {
            "image": image,
            "image_path": rel_path,
        }

        if self.has_labels:
          raw_label = int(row["label"])        
          label = raw_label - 1                
          sample["label"] = torch.tensor(label, dtype=torch.long)

          if self.return_attributes and self.attributes is not None:
              sample["attributes"] = torch.tensor(
                  self.attributes[label], dtype=torch.float32
              )


        if "id" in self.df.columns:
            sample["id"] = int(row["id"])

        return sample