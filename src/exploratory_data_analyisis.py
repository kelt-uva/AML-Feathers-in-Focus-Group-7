import pandas as pd
import matplotlib.pyplot as  plt

train_images_info = pd.read_csv("src/train_images.csv", index_col = 0)

def calculate_label_frequency(data):
    labels = data["label"]
    label_frequencies = pd.Series(labels).value_counts()
    frequency_barplot = label_frequencies.plot(kind = "bar")
    frequency_barplot.set_xlabel("")
    frequency_barplot.set_xticks([])
    plt.title("Distribution of label frequencies")
    plt.show()

calculate_label_frequency(train_images_info)

print("got labels")