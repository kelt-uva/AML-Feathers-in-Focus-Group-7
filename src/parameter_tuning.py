from train import main
import pandas as pd
from pathlib import Path
import os

def run():

    project_root = Path(__file__).resolve().parents[0]  

    save_path = project_root / f"./parameters/fitting_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    results = {}
    parameter_combinations = []
    accuracies = []
    f1_scores = []

    batch_sizes = [30]
    learning_rates = [1e-2]
    weight_decays = [1e-2]

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                print(f"Running training with batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}")
                acc, f1 = main(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay)
                parameter_combinations.append(f'batch_size = {batch_size}, learning_rate = {learning_rate}, weight_decay = {weight_decay}')
                accuracies.append(acc)
                f1_scores.append(f1)

    results['Parameter Combination'] = parameter_combinations
    results['Validation Accuracy'] = accuracies
    results['Validation F1 Score'] = f1_scores
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=True)

if __name__ == '__main__':
    run()