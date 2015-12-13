#!/bin/bash

source ~/Python/bin/activate
pkill mongod
mkdir /home/$USER/Research/Generative-Models-in-Classification/Results/Spearmint/HMM/ecogmixed
sleep 10
"/home/"$USER"/Software/mongodb/bin/mongod" --fork --logpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/HMM/ecogmixed/log.txt --dbpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/HMM/ecogmixed
cd ~/Software/Spearmint/spearmint
./cleanup.sh ~/Research/Generative-Models-in-Classification/Code/HMM/spearmint/ecogmixed/
python main.py ~/Research/Generative-Models-in-Classification/Code/HMM/spearmint/ecogmixed/

