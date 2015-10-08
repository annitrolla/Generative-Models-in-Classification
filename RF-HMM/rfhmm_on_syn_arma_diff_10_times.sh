#!/bin/bash

for i in {1..9}
do
    echo " "
    echo "-------------- Run $i --------------"
    python "$HOME/Research/Generative-Models-in-Classification/Code/DataNexus/gensyn_arma_diff_feature.py"
    python "$HOME/Research/Generative-Models-in-Classification/Code/RF-HMM/rfhmm_on_syn_arma_diff.py"
done
