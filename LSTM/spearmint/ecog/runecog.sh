#!/bin/bash

source ~/Python/bin/activate
source ~/.bash_profile
pkill mongod
sleep 10
"/home/"$USER"/Software/mongodb/bin/mongod" --fork --logpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/ecog/log.txt --dbpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/ecog
cd ~/Software/Spearmint/spearmint
./cleanup.sh ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/ecog/
python main.py ~/Research/Generative-Models-in-Classification/Code/LSTM/spearmint/ecog/

