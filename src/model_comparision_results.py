import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

comparision_results = pd.read_csv("./model_comparision/model_results.csv")

models = ["Model with dropout & bottleneck", "Simple model", "Deep and wide model", "Simplified VGG"]
validation_accuracy = comparision_results["Validation Accuracy"]
training_accuracy = comparision_results["Training Accuracy"]

num_models = range(len(models))

plt.figure(figsize = (12, 8))
plt.grid(axis = "y", color = "lightgrey", linestyle = "--", alpha = 0.5)
plt.scatter(num_models, validation_accuracy, color = "blue", label = "Validation Accuray")
plt.scatter(num_models, training_accuracy, color = "red", label = "Training Accuracy")
plt.xticks(num_models, models)
plt.ylabel("Accuracy")
plt.xlabel("Models")
plt.title("Model comparision based on accuracy")
plt.legend()
plt.savefig("./model_comparision/model_results.png")