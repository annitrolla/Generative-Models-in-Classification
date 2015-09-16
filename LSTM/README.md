LSTM
====

Initial run on LSTM with generative models gave 0.53 accuracy on the test set.


### Spearmint for Generative LSTM

##### size: 2000-10000
Running...

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


### Spearmint for Discriminative LSTM

##### size: 2000-10000
Running...

##### size: 50-500
See `spearmint/ecog/history/outputoutput-16.09.2015` for details  
```
Minimum expected objective value under model is -0.60345 (+/- 0.00666), at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                optim         enum       rmsprop
                lstmsize      int        254
                dropout       float      0.800000

Minimum of observed values is -0.624572, at location:
                NAME          TYPE       VALUE
                ----          ----       -----
                optim         enum       rmsprop
                lstmsize      int        359
                dropout       float      0.800000
```
