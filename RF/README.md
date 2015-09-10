# Random Forest
### Running
In order to run random forest do the following:  
`python rf_on_ECoG.py`
### Parameters
500 decision trees were chosen for this setup. The rest of the parameters left by default.
#### Data
The input data is sliced by 300 ms and transformed via Fourier.
#### Result
The achieved result on a training *100%* and on a test set - *0.63679*
