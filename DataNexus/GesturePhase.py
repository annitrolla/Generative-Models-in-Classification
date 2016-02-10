import numpy as np
import pandas as pd

for subject in ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']:
    raw = pd.read_csv('/storage/hpc_anna/GMiC/Data/GesturePhase/raw/%s_raw.csv' % subject, sep=',')
    prc = pd.read_csv('/storage/hpc_anna/GMiC/Data/GesturePhase/raw/%s_va3.csv' % subject, sep=',')

    # glue features together
    data = pd.concat([prc, raw], axis=1)
    
    # keep only Rest and Stroke classes
    raw = raw.loc[raw['phase'].isin(['Rest', 'Stroke'])]
    prc = prc.loc[raw['phase'].isin(['Rest', 'Stroke'])]
    
    # drop timestamp
    raw = raw.drop('timestamp', 1)
    prc = prc.drop('timestamp', 1)

    # replace class labels with numbers
    raw.



