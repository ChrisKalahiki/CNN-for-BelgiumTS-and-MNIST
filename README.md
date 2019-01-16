# CNN for BelgiumTS and MNIST
This repository contains a CNN or convolutional neural network in Python with the intention of identifying the 62 categories of street signs in the BelgiumTS data set. As a bonus, a second CNN was added to this repository that can be used for the classic MNIST dataset.

## The CNN Model
### The Input Layer
'''input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])'''
In this layer, we reshape X to a 4-D tensor containing the batch size, height, width, and number of channels.

### The First Convolutional Layer
'''conv1 = tf.layers.conv2d(
	inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)'''
In this layer, we compute 32 features using a 5x5 filter with the ReLU activation function. 
Padding is added to preserve width and height. The input tensor shape is [batch_size, 64, 64, 3].
The output tensor shape is [batch_size, 64, 64, 32].

### The First Pooling Layer
'''pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)'''
This layer computes 64 features using a 5x5 filter and a stride of 2. 
The input tensor shape is [batch_size, 64, 64, 32].
The output tensor shape is [batch_size, 32, 32, 64].

### The Second Convolutional Layer
'''conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)'''
In this layer, we compute 64 features using a 5x5 filter. Padding is added to preserve height and width.
The input tensor shape is now [batch_size, 32, 32, 32] and the output tensor shape is now {batch_size, 32, 32, 64]

### The Second Pooling Layer

### Flattening Tensor into a Batch of Vectors

### Dense Layer

### Dropout Operation

### Logits Layer

### Predictions

### Calculating Loss

### Training Operation

### Evaluation Metric

## Training the Model

## Testing the Model

## Loading a Model

## Clearing a Model

## The GUI

## Other Functions



## _*More Documentation Coming Soon*_
