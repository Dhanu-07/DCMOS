import numpy as np
import cslackandlpath
import dttldtodt
import gatedelays_softmin
import gatedelays
import jacobianut
import jacobianux
import lsgssolver
import nlbs_softmax
import nlbs
import timingasg

from cslackandlpath import *
from dttldtodt import *
from gatedelays_softmin import *
from gatedelays import *
from jacobianut import *
from jacobianux import *
from lsgssolver import *
from nlbs_softmax import *
from nlbs import *
from timingasg import *
import numpy as np
from scipy.sparse import csr_matrix

def timingasg(n, F, Fi, Fj, d):
    t = np.zeros(n)
    
    for i in range(len(Fj) - 1):

        t[i] = 0
        k = 0
        while Fj[i+1] - Fj[i] - k > 0:
            j = Fj[i] + k
            fig = Fi[j]
            fig = min(fig, len(t) - 1)
            t[i] = max(t[i], t[fig])
            k += 1
        t[i] += d[i]

    return t

def mexFunction(F,d,t):
    # Extract the inputs
    # F = prhs[0]
    # d = prhs[1]
    # t = prhs[2]
    
    Fi = F.indptr
    Fj = F.indices
    n = F.shape[0]

    t[:] = timingasg(n, F, Fi, Fj, d)

    return t

# Example usage:
# Assume F is a sparse matrix, and d, t are NumPy arrays
# Here's an example to demonstrate:
# from scipy.sparse import csr_matrix

# # Example sparse matrix F (3x3 matrix for demonstration)
# data = np.array([1, 1, 1])
# rows = np.array([0, 1, 2])
# cols = np.array([0, 1, 2])
# F = csr_matrix((data, (rows, cols)), shape=(3, 3))

# # Example d array (the durations or other parameters)
# d = np.array([1.0, 2.0, 3.0])

# # Initialize t array to hold the result
# t = np.zeros(3)

# # Now, use mexFunction to compute t
# result = mexFunction([F, d, t])
# print(result)


