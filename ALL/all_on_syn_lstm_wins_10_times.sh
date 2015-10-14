#!/bin/bash

for i in {1..9}
do
    echo " "
    echo "-------------- Run $i --------------"
    python "$HOME/Research/Generative-Models-in-Classification/Code/DataNexus/gensyn_lstm_wins.py"
    python "$HOME/Research/Generative-Models-in-Classification/Code/ALL/all_on_syn_lstm_wins.py"
done
