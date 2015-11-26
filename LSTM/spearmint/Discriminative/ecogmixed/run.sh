#!/bin/bash

source ~/Python/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu,floatX=float32'
pkill mongod
sleep 10
"/home/"$USER"/Software/mongodb/bin/mongod" --fork --logpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/LSTM/Discriminative/ecogmixedlog.txt --dbpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/LSTM/Discriminative/ecogmixed
cd ~/Software/Spearmint/spearmint
./cleanup.sh ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/Discriminative/ecogmixed/
python main.py ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/Discriminative/ecogmixed/

