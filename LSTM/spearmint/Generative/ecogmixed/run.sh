#!/bin/bash

source ~/Python/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu,floatX=float32'
mkdir /home/$USER/Research/Generative-Models-in-Classification/Results/Spearmint/LSTM/Generative/ecogmixed
pkill mongod
sleep 10
"/home/"$USER"/Software/mongodb/bin/mongod" --fork --logpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/LSTM/Generative/ecogmixed/log.txt --dbpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/LSTM/Generative/ecogmixed
cd ~/Software/Spearmint/spearmint
./cleanup.sh ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/Generative/ecogmixed/
python main.py ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/Generative/ecogmixed/

