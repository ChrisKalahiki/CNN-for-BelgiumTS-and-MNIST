# CNN for BelgiumTS and MNIST
This repository contains a CNN or convolutional neural network in Python with the intention of identifying the 62 categories of street signs in the BelgiumTS data set. As a bonus, a second CNN was added to this repository that can be used for the classic MNIST dataset.

## Requirements
This program requires Python 3. It will also require the following Python packages:
numpy, tensorflow, shutil, skimage, matplotlib, tkinter

## Getting Started
There are two versions of the script for running the model on the BelgiumTS dataset. The first includes a GUI and can be run using the following command:
`python3 src/cnn_RMR_GUI.py`

You can also run the model without the GUI using the following command:
`python3 src/cnn_RMR.py`

There is also a script for running the model on MNIST that is located at:
`python3 src/cnn_mnist.py`

## Data
The complete datasets are included in the Data folder. The MNIST data is still compressed, but the BelgiumTS dataset is unzipped and ready to go by default. The data is alreay split into training and testing sets and further divided into folders for each image type.