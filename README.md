# AML-Feathers-in-Focus-Group-7

This repository contains contributions to the Applied Machine Learning group project; Feathers in Focus. Below, we explain the directory structure and file purposes.

## Notebooks

The `/notebooks` directory contains three notebooks which were used in the initial stages of the project. `data_imports.ipynb` reads and stores all image data in `.pkl` format in the `/pickles` directory for efficient usage. `baseline_predictions.ipynb` uses the `Birds-Classifier-EfficientNetB2` model from Hugging Face to create baseline predictions, and stores these in the `/predictions` directory. `initial_visualisations.ipynb` plots some randomly selected samples and a random convolution to get an idea of what the data looks like and what kinds of transformations a CNN could apply.

## Source

The `/src` directory contains all the working files for the project, as well as the source data, test results, and figures.

`exploratory_data_analysis.py` plots the class distribution in the data, revealing a strong class imbalance.

`dataset.py` creates a module used for importing the image data (making the `.pkl` files created earlier obsolete). It contains a transforms function which allows standardisation and augmentation.

`model.py` contains all models we used in our project, including our custom CNN, `efficientnet-b0`, and some others.

`train.py` is the primary file in which everything comes together. It imports a model from `model.py` and uses the module created in `dataset.py` to read and prepare the data. It then trains and validates the chosen model using a standard pipeline. The best model is saved as a `.pth` file. Some figures, susch as training curves, are saved in `/vis`.

`predictions.py` loads a best model from a `.pth` file, and performs predictions on the test images, saving them in `/preds` as a `.csv` ready for Kaggle submission.

`parameter_tuning.py` performs a gridsearch-like approach to hyperparameter tuning, iterating over all given combinations, and saving all best model outcomes (validation accuracy, F1, etc.) as `fitting_results.csv` in `/parameters`. The 'optimal' set of hyperparameters was set as the default values in `train.py`.

`layer_vis.py` loads the best custom model and uses a randomly selected image to exemplify the model's transformations by extracting the output of each block/layer in the custom CNN, and saving two random channels per block in `/example`.

`run_models.py` trains one instance of a selection of models in `model.py` and stores the results in `/model_comparision/model_results.csv`. These results are then plotted by `model_comparison_results.py`, and also stored in `/model_comparison`.

`cross_val.py` performs a k-fold cross validation on the custom CNN, and computes and plots the performance metrics in a grouped barchart (`/vis/cv_folds_comparison.png`).

`check_predicted_labels.py` performs a brief analysis of the most frequently predicted labels (both with and without weighted sampling).

## References

“Introduction to Deep Learning with PyTorch,” Datacamp.com, Dec. 12, 2025. https://www.datacamp.com/courses/introduction-to-deep-learning-with-pytorch (accessed Dec. 14, 2025).

“Intermediate Deep Learning with PyTorch,” Datacamp.com, Jun. 19, 2025. https://www.datacamp.com/courses/intermediate-deep-learning-with-pytorch (accessed Dec. 14, 2025).

“Deep Learning for Images with PyTorch,” Datacamp.com, Jun. 25, 2025. https://www.datacamp.com/courses/deep-learning-for-images-with-pytorch (accessed Dec. 14, 2025).
‌
‌“Python Convolutional Neural Networks (CNN) with TensorFlow Tutorial,” www.datacamp.com. https://www.datacamp.com/tutorial/cnn-tensorflow-python

N. Adaloglou, “Best deep CNN architectures and their principles: from AlexNet to EfficientNet,” AI Summer, Jan. 21, 2021. https://theaisummer.com/cnn-architectures/

C. Hughes, “Demystifying PyTorch’s WeightedRandomSampler by example,” Medium, Aug. 30, 2022. https://medium.com/data-science/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452 (accessed Oct. 17, 2025).

“torchvision.transforms — Torchvision master documentation,” Pytorch.org, 2017. https://docs.pytorch.org/vision/0.9/transforms.html

PyTorch Contributors, “torch.topk,” Pytorch.org, 2023. https://docs.pytorch.org/docs/stable/generated/torch.topk.html (accessed Dec. 14, 2025).

Oreolorun Olu-Ipinlaye, “How Convolutional Autoencoders Power Deep Learning Applications,” Digitalocean.com, Oct. 14, 2022. https://www.digitalocean.com/community/tutorials/convolutional-autoencoder#bottleneck-and-details (accessed Dec. 14, 2025).