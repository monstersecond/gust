#!/bin/bash

# build temporary directories
if [ ! -d "./tmp_data" ];then
    mkdir ./tmp_data
fi
if [ ! -d "./tmp_img" ];then
    mkdir ./tmp_img
fi
if [ ! -d "./tmp_net" ];then
    mkdir ./tmp_net
fi
if [ ! -d "./results" ];then
    mkdir ./results
fi


# gain datasets
# the first parameter is the train object number in the dataset
# the second parameter is the test object number in the dataset
python ./dataset/shapes_multiobj.py 2 3
python ./dataset/shapes_multiobj.py 3 3
python ./dataset/shapes_multiobj.py 4 3

python ./dataset/shapes_multiobj.py 3 2
python ./dataset/shapes_multiobj.py 3 4

# train network
python train.py
