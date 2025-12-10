from train import main
import pandas as pd
from pathlib import Path
import os

def run():

    project_root = Path(__file__).resolve().parents[0]  

    save_path = project_root / f"./parameters/fitting_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    results = {}
    parameter_batch_size = []
    parameter_learning_rate = []
    parameter_weight_decay = []
    accuracies = []
    f1_scores = []
    train_accuracies = []
    train_f1_scores = []

    batch_sizes = [32]
    learning_rates = [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    #learning_rates = [5e-5]
    weight_decays = [1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    weight_decays = [1e-2]
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                print(f"Running training with batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}")
                train_acc, train_f1, acc, f1 = main(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay)
                parameter_batch_size.append(batch_size),
                parameter_learning_rate.append(learning_rate),
                parameter_weight_decay.append(weight_decay),
                accuracies.append(acc)
                f1_scores.append(f1)
                train_accuracies.append(train_acc)
                train_f1_scores.append(train_f1)

    results['Batch size'] = parameter_batch_size
    results['Learning rate'] = parameter_learning_rate
    results['Weight decay'] = parameter_weight_decay
    results['Validation Accuracy'] = accuracies
    results['Validation F1 Score'] = f1_scores
    results['Training Accuracy'] = train_accuracies
    results['Training F1 Score'] = train_f1_scores
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=True)

if __name__ == '__main__':
    run()