import torch
from dataset import get_transforms, BirdDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from model import create_own_model
import os
import time

def set_seed(seed = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fetch_data(test_csv_path, image_size = 224):
    test_dataset = BirdDataset(csv_path=test_csv_path, transform=get_transforms(False, image_size=image_size), has_labels=False)
    return test_dataset

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted

def main():

    set_seed(42)

    project_root = Path(__file__).resolve().parents[0]  
    test_csv_path = project_root / "test_images_path.csv"

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = project_root / f"./preds/predictions{timestamp}.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    print("Using device:", device)

    model = create_own_model()
    model.to(device)
    model.load_state_dict(torch.load(project_root / "best_custom_model.pth"))
    model.eval()

    print("Model loaded and set to evaluation mode.")

    test_dataset = fetch_data(test_csv_path=test_csv_path, image_size=224)

    test_dataset_lentgh = test_dataset.__len__()

    prediction_labels = []
    prediction_ids = []
    print("Starting predictions on test dataset...")
    for num, i in enumerate(test_dataset):
        print(f'Predicting images... {int(((num + 1) / test_dataset_lentgh) * 100)}% complete{'!' if num + 1 == test_dataset_lentgh else '.'}', end='\r')

        image = i['image'].unsqueeze(0).to(device)
        image_id = i['id']

        prediction = predict(model, image).item()
        prediction_labels.append(prediction)
        prediction_ids.append(image_id)

        if num == test_dataset_lentgh - 1:
            print('Predictions complete.')
    
    prediction_df = pd.DataFrame({'id':prediction_ids, 'label':prediction_labels})

    prediction_df.to_csv(save_path, index = False)

    print(f"Predictions saved to /preds/predictions{timestamp}.csv. Prediction dataframe shape: {prediction_df.shape}.")

if __name__ == "__main__":
    main()