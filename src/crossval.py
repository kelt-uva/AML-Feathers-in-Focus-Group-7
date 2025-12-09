from train import main
import pandas as pd

def run():

    results = {}

    batch_sizes = [64, 128, 256]
    learning_rates = [1e-3, 2e-4, 1e-5]
    weight_decays = [1e-2, 1e-3, 1e-4]
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                print(f"Running training with batch_size={batch_size}, learning_rate={learning_rate}, weight_decay={weight_decay}")
                acc, f1 = main(batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay)
                results[f'batch_size = {batch_size}, learning_rate = {learning_rate}, weight_decay = {weight_decay}'] = f'Best validation accuracy = {acc}, best f1 score = {f1}'
    results_df = pd.DataFrame(results)
    results_df.to_csv('./crossval/crossval_results.csv', index=True)

if __name__ == '__main__':
    run()