import pandas as pd
import matplotlib.pyplot as  plt
import os

train_images_info = pd.read_csv("src/train_images.csv", index_col = 0)
save_path_frequencies = f"./vis/label_frequencies.png"

def calculate_label_frequency(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok = True)
    labels = data["label"]
    label_frequencies = pd.Series(labels).value_counts()
    fig, ax = plt.subplots(figsize = (12, 8))
    label_frequencies.plot(kind = "bar", ax = ax)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of label frequencies")
    fig.savefig(save_path)
    print(f"Label frequency figure saved to {save_path}")
    plt.show()
    plt.close(fig)

calculate_label_frequency(train_images_info, save_path_frequencies)