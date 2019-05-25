# Repository: fashion_mnist 

_This is a work in progress (WIP), and this sentence will be removed once it's released._

This repository consist of two sub projects...

..* Tensorflow 2 Alpha
..* Docker Image
..* Docker File
..* Fashion MNIST Dataset


## Train

`docker build -t adamowsley/fashion_train:0.1 .`

`docker run -u $(id -u):$(id -g) --runtime=nvidia -it --rm -v $PWD:/opt/app adamowsley/fashion_train:0.1`



## Predict

