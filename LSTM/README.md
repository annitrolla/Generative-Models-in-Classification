LSTM
====

Initial run on LSTM with generative models gave 0.53 accuracy on the test set.


### Spearmint for Generative LSTM

##### size: 50-2000
See `spearmint/ecogbin/history/outputoutput-16.09.2015` for details  
```
Minimum expected objective value under model is -0.76885 (+/- 0.01627), at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                optim         enum       adagrad
                lstmsize      int        1991
                dropout       float      0.800000

Minimum of observed values is -0.799658, at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                optim         enum       adagrad
                lstmsize      int        2000
                dropout       float      0.478501
```
