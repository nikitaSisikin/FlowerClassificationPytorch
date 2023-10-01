# Flower Image Classifier

This repository contains a Python application for training a flower image classifier using PyTorch and transfer learning. The trained model can be used for predicting the class of flower images. 

# Prerequisites

Make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision

You can install the required packages by running:

```
conda install --file requirements.txt
```

# Dataset

The dataset used for training the model is available [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

Download the dataset and organize it into train, valid, and test sets within a directory.

Or, download the orginized dataset [here](https://file.io/rWfnLNDkEKUv).




![flowers](https://github.com/nikitaSisikin/FlowerClassificationPytorch/assets/74993680/cd1ec250-473c-403c-a853-633734b6236c)



# Training the Model

To train the flower image classifier, use the train.py script. Here's an example command:

```
python train.py data_dir --save_dir checkpoints --arch resnet18 --learning_rate 0.001 --hidden_units 1024 --epochs 10 --gpu
```

+ **data_dir:** Path to the data directory containing train, valid, and test sets.
+ **--save_dir:** Directory to save checkpoints (default is 'checkpoints').
+ **--arch:** Architecture name (e.g., 'resnet18' or 'densenet121').
+ **--learning_rate:** Learning rate for training (default is 0.001).
+ **--hidden_units:** Number of hidden units in the classifier (default is 1024).
+ **--epochs:** Number of training epochs (default is 10).
+ **--gpu:** Flag to enable GPU training.
  
The trained model will be saved in the specified directory.

# Predicting Flower Images

To predict the class of a flower image, use the predict.py script. Here's an example command:

```
python predict.py image_path checkpoint.pth --top_k 3 --category_names categories.json --gpu
```

+ **image_path:** Path to the image file for prediction.
+ **checkpoint.pth:** Path to the model checkpoint file.
+ **--top_k:** Number of top classes to display (default is 1).
+ **--category_names:** Path to JSON file mapping categories to names.
+ **--gpu:** Flag to use GPU for inference.
  
# Repository Structure

+ **train.py:** Script for training the flower image classifier.
+ **predict.py:** Script for predicting the class of a flower image using a trained model.
+ **json_util.py:** Utility module for loading category names from a JSON file.
+ **categories.json:** JSON file mapping category numbers to category names.
+ **requirements.txt:** List of required packages for the Conda environment.
