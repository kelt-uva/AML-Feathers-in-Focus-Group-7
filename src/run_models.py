from train import main
import pandas as pd
from pathlib import Path
import os
from model import create_own_model, create_simple_model, create_deep_wide, create_vgg

def run():

    save_path = "./model_comparision/model_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    model_own = create_own_model()
    model_simple = create_simple_model()
    model_deep_wide = create_deep_wide()
    model_vgg = create_vgg()

    models = [["model_own", model_own], ["model_simple", model_simple], ["model_deep_wide", model_deep_wide], ["model_vgg", model_vgg]]

    results = {}
    run_models = []
    accuracies = []
    f1_scores = []
    train_accuracies = []
    train_f1_scores = []



    for model in models:
        print(f"Running training with model = {model[0]}")
        train_acc, train_f1, acc, f1 = main(batch_size=32, num_epochs=5, learning_rate=1e-4, weight_decay=1e-3, tuning=False, model = model[1])
        run_models.append(model[0])
        accuracies.append(acc)
        f1_scores.append(f1)
        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)

    results['Model'] = run_models
    results['Validation Accuracy'] = accuracies
    results['Validation F1 Score'] = f1_scores
    results['Training Accuracy'] = train_accuracies
    results['Training F1 Score'] = train_f1_scores
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=True)

if __name__ == '__main__':
    run()