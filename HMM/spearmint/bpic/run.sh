#!/bin/bash

source ~/Python/bin/activate
pkill mongod
sleep 10
"/home/"$USER"/Software/mongodb/bin/mongod" --fork --logpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/HMM/bpic/log.txt --dbpath ~/Research/Generative-Models-in-Classification/Results/Spearmint/HMM/bpic
cd ~/Software/Spearmint/spearmint
./cleanup.sh ~/Research/Generative-Models-in-Classification/Code/HMM/spearmint/bpic/
python main.py ~/Research/Generative-Models-in-Classification/Code/HMM/spearmint/bpic/

