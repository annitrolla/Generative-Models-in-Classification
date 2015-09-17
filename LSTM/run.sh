#!/bin/bash

source ~/Python/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu,floatX=float32'
python lstm_classifier.py

