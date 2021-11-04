#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#herbert sanford, chris kalahiki, spencer graff
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import shutil
import skimage.data
import skimage.transform
import random
import matplotlib
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Our images are 64x64 pixels, and have 3 color channels
    input_layer = tf.reshape(features["x"], [-1, 64, 64, 3])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 64, 64, 3]
    # Output Tensor Shape: [batch_size, 64, 64, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 64, 64, 32]
    # Output Tensor Shape: [batch_size, 32, 32, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 32, 32, 32]
    # Output Tensor Shape: [batch_size, 32, 32, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 32, 32, 64]
    # Output Tensor Shape: [batch_size, 16, 16, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 16, 16, 64]
    # Output Tensor Shape: [batch_size, 16 * 16 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 16 * 16 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 62]
    logits = tf.layers.dense(inputs=dropout, units=62)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=62)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def getData(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)

                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return np.array(resizeImages(images),dtype=np.float32), np.array(labels,dtype=np.int32)

def resizeImages(images):
    reimages = [skimage.transform.resize(image, (64, 64), mode='reflect') for image in images]
    return reimages

def train_network(classifier, logging_hook):
    train_data, train_labels = getData('../Data/BelgiumTS/Training')

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=100,  # 20000
        hooks=[logging_hook])

def test_network(classifier):
    eval_data, eval_labels = getData('../Data/BelgiumTS/Testing')

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    printData(eval_results)

def load_network():
    return tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

def clear_network():
    shutil.rmtree("/tmp/mnist_convnet_model")

def display_test():
    images, labels = getData('../Data/BelgiumTS/Testing')
    # Pick 10 random images
    sample_indexes = random.sample(range(len(images)), 10)
    sample_images = [images[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]

    # Run the "predicted_labels" op.
    session = tf.Session()
    images_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])
    images_flat = tf.contrib.layers.flatten(images_ph)
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)
    predicted_labels = tf.argmax(logits, 1)
    predicted = session.run([predicted_labels], feed_dict={images_ph: sample_images})[0]
    print(sample_labels)
    print(predicted)

    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1 + i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                 fontsize=12, color=color)
        plt.imshow(sample_images[i])

def revNetwork():
    # Create the Estimator
    classifier = load_network()

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    #train_network(classifier, logging_hook)
    test_network(classifier)
    #display_test()

def warmstartNetwork():
    # Load training and eval data
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    #train_data = mnist.train.images  # Returns np.array
    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    #eval_data = mnist.test.images  # Returns np.array
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    train_data, train_labels = getData('../Data/BelgiumTS/Training')
    eval_data, eval_labels = getData('../Data/BelgiumTS/Testing')


    # Create the Estimator
    mnist_classifier = load_network()

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook])

    # Evaluate the model and save results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    saveData(eval_results)
    printData(eval_results)

# This section will change to the new data set
def coldstartNetwork():
    # Load training and eval data
    #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    #train_data = mnist.train.images  # Returns np.array
    #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    #eval_data = mnist.test.images  # Returns np.array
    #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    train_data, train_labels = getData('../Data/BelgiumTS/Training')
    eval_data, eval_labels = getData('../Data/BelgiumTS/Testing')


    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=10,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook])

    # Evaluate the model and save results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    saveData(eval_results)
    printData(eval_results)

def saveData(data):
    my_file = 'out/results.txt'
    deleteOldResults()
    if os.path.isfile(my_file):
        with open(my_file, 'r+') as f:
            f.seek(0)
            f.truncate()
            f.write(str(data))
            f.close()
    else:
        open(my_file, 'w+')
        my_file.write(str(data))
        my_file.close()

def deleteOldResults():
    my_file = 'out/results.txt'
    if os.path.isfile(my_file):
        with open(my_file, 'r+') as f:
            f.seek(0)
            f.truncate()
            f.seek(0)
            f.write(" ")
            f.close()
    else:
        f = open(my_file, 'w+')
        my_file.write(" ")
        my_file.close()

def printData(data):
    window = Tk()
    window.title("Your Results Are")
    window.configure(background="white")
    txt = Label(window, text=(str(data)))
    txt.configure(background="#00FFFF", fg="black", font="bold")
    txt.pack(padx=50, pady=50)

def displayData():
    my_file = 'out/results.txt'
    if os.path.isfile(my_file):
        with open(my_file, 'r') as f:
           first_line = f.readline()
           f.close()
           printData(first_line)
    else:
        printData("No Data to Show")

def mainWindow():

    #root
    root = Tk()
    root.title("Neural Network RMR")
    root.configure(background='black')
    root.configure(background='white')

    # button setup
    frame = Frame(root)
    frame.configure(background='white')
    frame.pack(side=LEFT, padx=10, pady=10)
    buttonWidth = 25

    # buttons
    buttonTrain = Button(frame, text="Train Nework", command=lambda: coldstartNetwork(), width=buttonWidth)
    buttonDisplay = Button(frame, text="Display Most Recent Results", command=lambda: displayData(), width=buttonWidth)
    buttonTest = Button(frame, text="Test Network", command=lambda: revNetwork(), width=buttonWidth)
    buttonContinue = Button(frame, text="Continue Train Network", command=lambda: warmstartNetwork(), width=buttonWidth)

    # button config
    buttonTrain.configure(background="#ff1111", fg="black", font="Bold")
    buttonDisplay.configure(background="#00FFFF", fg="black", font="Bold")
    buttonTest.configure(background="#228b22", fg="black", font="Bold")
    buttonContinue.configure(background="#228b22", fg="black", font="Bold")

    #button allign
    buttonTrain.grid(row=0, column=0)
    buttonDisplay.grid(row=1, column=0)
    buttonTest.grid(row=1, column=1)
    buttonContinue.grid(row=0, column=1)

    root.mainloop()

mainWindow()