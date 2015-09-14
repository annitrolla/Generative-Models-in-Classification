#!/bin/bash

source ~/Python/bin/activate
source ~/.bash_profile
pkill mongod
sleep 10
/home/hpc_kuz/Software/mongodb/mongodb-linux-x86_64-3.0.4/bin/mongod --fork --logpath ~/Research/Generative-Models-in-Classification/Results/ecog/log.txt --dbpath ~/Research/Generative-Models-in-Classification/Results/ecog
cd ~/Software/Spearmint/spearmint
./cleanup.sh ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/ecog/
python main.py ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/ecog/

