#!/bin/sh
STRING="Downloading the MNIST-data set and creating clients"
echo $STRING
eval mkdir MNIST_original
eval cd MNIST_original 
eval curl -O "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
eval curl -O "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
eval curl -O "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
eval curl -O "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
eval gunzip train-images-idx3-ubyte.gz
eval gunzip train-labels-idx1-ubyte.gz
eval gunzip t10k-images-idx3-ubyte.gz
eval gunzip t10k-labels-idx1-ubyte.gz
eval cd ..
eval python Create_clients.py 

