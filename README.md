# Repository: fashion_mnist 

TensorFlow is an open-source software library used for analysis of big data and machine learning 
applications. One such application is for predicting or classifying images using Convolutional
Neural Networks (CNN), and this project focuses on this area.

The images are supplied by the Modified National Institute of Standards and Technology (MNIST)
database. The fashion database contains 60,000 training images and 10,000 test and evaluation 
images.

This code repository itself consists of two sub projects that use Docker images for all processing, and each 
sub project consists of or uses the following items:

1. tensorflow/tensorflow:2.0.0a0-gpu-py3 Docker image
2. Dockerfile that details our development environment
3. requirements.txt that details dependencies for our development environment
4. Fashion MNIST Dataset

As you can see, we are using a Docker image for this, so there is no need to install TensorFlow
on your host machine.

### Operating System

This code was developed and tested on Ubuntu 18.04.2 LTS and an NVIDIA GeForce GTX graphics card. 
You may need to adjust the TensorFlow docker image in the `Dockerfile` based on your system 
configuration.

### Running Docker

Before I begin using Docker, I execute the following two commands. This allows Docker to access 
the NVIDIA runtime. You may not face a similar issue.

1. sudo systemctl daemon-reload
2. sudo systemctl restart docker

## Train

This project uses the fashion MNIST dataset to train a Convolutional Neural Network model that is 
saved to a disk image. For my purposes, I then use this model in the `predict` project to predict or 
classify an input image. Please see comments in the `fashion_train.py` file for additional information.

#### Build Docker Image

First, we need to build a docker image. This file defines our environment, and you may change it to 
suit your own system. 

The most important part is the specification of the TensorFlow Alpha image. I prefer most things new, 
so I'm using the TensorFlow 2 Alpha image that includes Python3 and GPU processing.

`docker build -t adamowsley/fashion_train:0.1 .`

The following statement trains our model. This statement executes in your current working directory.
When it is completed, you will find a newly created `trained_model.h5` file.

Once you are satisfied with the training and loss efficiencies, copy or move the `trained_model.h5`
file to the `predict` directory.

`docker run -u $(id -u):$(id -g) --runtime=nvidia -it --rm -v $PWD:/opt/app adamowsley/fashion_train:0.1`

## Predict

Build the `fashion_predict` docker image. This file defines our environment just as before, 
and you may change it to suit your own system configuration. 

`docker build -t adamowsley/fashion_predict:0.1 .`

Use the following `docker run` statement to make your prediction. Use small images containing 
solid light colored backgrounds for testing your predictions as the MNIST training set has 
similar properties. 

`docker run -u $(id -u):$(id -g) --runtime=nvidia -it --rm -v $PWD:/opt/app adamowsley/fashion_predict:0.1 <file_name>`

Once the prediction code runs to completion, you will find two images in the directory along with 
a probability array that is displayed to your console. The two images are the `prediction.png` and 
`small.png` image files.

#### prediction.png

This image displays a bar graph of each class and the probability that our input image belongs to a
particular class. Try using large images with non solid backgrounds for fun.

#### small.png

This image is the scaled version of your input image. Remember that our training set images are
28 x 28 grayscale images. You will see that very large images don't scale well to such a small input
shape.

# Conclusions

It is my opinion that training images need to be larger for any practical application. As very 
large images don't seem to scale to very small sizes very well. Conjecture leads me to believe that 
training images should be somewhere in the range of 128 to 256 pixels square. The next step for me 
may be to locate such a training set. The following URL may be a good place to start the
search.

`https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research`