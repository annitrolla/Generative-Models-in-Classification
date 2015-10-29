"""

Generates parameter search grid from the list of paramteres and
stores it to file for bash script to pick up

"""

import numpy as np
import itertools

def expandgrid(*itrs):
    product = list(itertools.product(*itrs))
    return [[x[i] for x in product] for i in range(len(itrs))]

nsamples = [500, 1000, 1500, 2000, 2500, 3000]
nfeatures = [1, 5, 10, 20, 30]
nseqfeatures = [1, 5, 10, 20, 30]
seqlen = [3, 10, 30, 50]

prs = np.array(expandgrid(nsamples, nfeatures, nseqfeatures, seqlen)).T
with open('../../Results/grid.txt', 'w') as f:
    for p in prs:
        f.write('%d, %d, %d, %d\n' % (p[0], p[1], p[2], p[3]))
f.close()

