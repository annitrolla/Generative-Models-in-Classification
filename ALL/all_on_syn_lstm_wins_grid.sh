#!/bin/bash

RUN=0
while IFS=, read nsamples nfeatures nseqfeatures seqlen
do
    echo "------------------------- Run $RUN -------------------------"
    python all_on_syn_lstm_wins_grid.py -n $nsamples -s $nfeatures -d $nseqfeatures -l $seqlen
    let RUN=RUN+1
done < ../../Results/grid.txt
