#!/bin/bash

for i in {1..9}
do
    echo " "
    echo "-------------- Run $i --------------"
    python "$HOME/Research/Generative-Models-in-Classification/Code/ALL/all_on_ecogmixed.py"
done
