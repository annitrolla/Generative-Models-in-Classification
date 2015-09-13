# Hidden Markov Models (HMM)
Hidden markov models are generative models that are fit separately for each of the classes.

### Running
In order to run HMM do the following:  `python hmm_on_ECoG.py`  
Note that HMM can be performed with two different options:  
a) by activating function `hmmcl.find_best_parameter(ratio, hdn_nstates_list, niter, nrepetitions)`, which uses grid search over pre-specified number of hidden states  
b) by activating function `hmmcl.test_model(nstates, niter)`, which tests the model with the tuned parameters on a test set.

### Parameters
In case *a)* the parameters specified are
* **ratio** - splitting ratio for train and test data. In this setup it is 70% for training and 30% for testing
* **hdn_nstates_list** - range of hidden parameters to look for, currently range(2,21)
* **niter** - number of iterations to perform for training if it is not converged, set to 10
* **nrepetitions** - number of times the experiment should be repeated, now set to 5
When *b)* is chosen the parameters that require specification are number of states and number of iterations  

### Data 
The input data are sliced raw sequences with 300 ms each.

### Results
Results for the option a) are stored in `../../Results/crossvalidated_accuracy.txt`. In the second case they are printed to a console. 
Current results are: 
```
2, 0.742294520548, 0.74529109589, 0.741438356164, 0.763270547945, 0.717465753425
3, 0.746147260274, 0.741438356164, 0.751284246575, 0.752568493151, 0.726455479452
4, 0.756421232877, 0.753424657534, 0.769691780822, 0.770976027397, 0.736301369863
5, 0.75470890411, 0.778253424658, 0.767979452055, 0.751712328767, 0.762842465753
6, 0.766695205479, 0.77654109589, 0.768835616438, 0.783390410959, 0.780393835616
7, 0.812928082192, 0.806078767123, 0.779965753425, 0.78125, 0.806506849315
8, 0.791952054795, 0.797089041096, 0.826198630137, 0.80522260274, 0.797945205479
9, 0.813356164384, 0.815068493151, 0.816352739726, 0.803938356164, 0.815068493151
10, 0.833904109589, 0.831335616438, 0.8125, 0.835188356164, 0.84375
11, 0.848030821918, 0.816352739726, 0.832191780822, 0.840753424658, 0.819349315068
12, 0.839897260274, 0.836044520548, 0.851883561644, 0.842037671233, 0.832191780822
13, 0.856592465753, 0.833476027397, 0.854452054795, 0.845462328767, 0.854452054795
14, 0.845462328767, 0.840325342466, 0.859160958904, 0.865154109589, 0.852311643836
15, 0.857448630137, 0.868150684932, 0.857448630137, 0.846746575342, 0.866866438356
16, 0.860017123288, 0.859589041096, 0.861729452055, 0.851455479452, 0.872859589041
17, 0.871575342466, 0.877568493151, 0.875428082192, 0.853167808219, 0.869863013699
18, 0.886558219178, 0.869863013699, 0.876284246575, 0.887842465753, 0.881849315068
19, 0.868150684932, 0.883561643836, 0.880993150685, 0.876284246575, 0.892123287671
20, 0.884417808219, 0.881849315068, 0.891267123288, 0.885702054795, 0.890839041096
```
