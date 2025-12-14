import pandas as pd
import matplotlib.pyplot as plt

predictions_vs_labels_normal_sampling = pd.read_csv("results/predictions_vs_labels_normal_sampling.csv", index_col = 0)
predictions_vs_labels_weighted_sampling = pd.read_csv("results/predictions_vs_labels_weighted_sampling.csv", index_col = 0)

predictions_normal_sampling = predictions_vs_labels_normal_sampling["Predictions"]
predictions_weighted_sampling = predictions_vs_labels_weighted_sampling["Predictions"]

prediction_counts_normal_sampling = predictions_normal_sampling.value_counts()
prediction_counts_weighted_sampling = predictions_weighted_sampling.value_counts()

print(len(prediction_counts_normal_sampling))
print(len(prediction_counts_weighted_sampling))

print(f"Most frequently predicted labels: {prediction_counts_normal_sampling.head()}")
print(f"Less frequently predicted labels: {prediction_counts_normal_sampling.tail()}")

train_images_info = pd.read_csv("src/train_images.csv", index_col = 0)
labels = train_images_info["label"]
label_frequencies = pd.Series(labels).value_counts()

print("Occurrence of most frequently predicted labels in training dataset")
print(label_frequencies.get(37, 0))
print(label_frequencies.get(85, 0))
print(label_frequencies.get(144, 0))
print(label_frequencies.get(32, 0))
print(label_frequencies.get(2, 0))

print("Occurrence of least frequently predicted labels in the dataset")
print(label_frequencies.get(71, 0))
print(label_frequencies.get(44, 0))
print(label_frequencies.get(157, 0))
print(label_frequencies.get(20, 0))
print(label_frequencies.get(160, 0))

print(label_frequencies.head())
print(label_frequencies.tail())

not_predicted_labels = label_frequencies.index.difference(prediction_counts_normal_sampling.index)
print(len(not_predicted_labels))
frequencies_not_predicted = label_frequencies[not_predicted_labels]

ax = frequencies_not_predicted.plot(kind = "bar")
plt.ylabel("Frequencies")
plt.xlabel("Label")
ax.set_xticks([])
ax.set_xticklabels([])
plt.title("Frequency of not predicted labels in the dataset")
plt.show()
