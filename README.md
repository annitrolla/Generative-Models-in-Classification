# Generative Models in Classification

### Basic Pipeline
<img src="Documentation/Figures/models_eleven.png" width="728px" />

### Folder structure
```
Project root
    ./Code   # THIS REPOSITORY
        ./Preprocessing  # scripts related to data preprocessing 
        ./RF             # usual classification using Random Forest
        ./HMM            # classification using HMM generative model
        ./LSTM           # classification using LSTM recurrent neual networks as a generative model
        ./RF-HMM         # hybrid classification: with HMM features
        ./RF-LSTM        # hybrid classification: with LSTM features
    ./Data   # datasets
    ./Lib    # third-party code
```

### Notes about running on HPC

The dafault way is  
`srun --partition=gpu --gres=gpu:1 --constraint=K20 --mem=20000 runecog.sh`  
however if you have one instance of mongodb running on one node, then you cannot run another one on the same node, so once you have initiated one run, use `squeue` to see which unit your task got allocated to and exclude it from the other run (all this matters is you run two Spearmint experiments in parallel). Say your first experiments got allocated to the node `idu38`, then your second experiment will be ran as  
`srun --exclude=idu38 --partition=gpu --gres=gpu:1 --constraint=K20 --mem=20000 runecogbin.sh`

### Tips & Tricks

###### Reload class in an interactive Python shell
```
import HMM.hmm_classifier
reload(HMM.hmm_classifier)
from HMM.hmm_classifier import HMMClassifier
hmmcl = HMMClassifier()
```

###### Execute before running GPU-dependent code
```
source ~/Python/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu,floatX=float32'
```
