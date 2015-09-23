#!/bin/bash

source ~/Python/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu,floatX=float32'
pkill mongod
sleep 10
"/home/"$USER"/Software/mongodb/bin/mongod" --fork --logpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/ecogbin/log.txt --dbpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/ecogbin
cd ~/Software/Spearmint/spearmint
./cleanup.sh ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/ecogbin/
python main.py ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/ecogbin/

