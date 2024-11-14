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
def jacobianut(n, F, Fi, Fj, dmin, u, t, p):
    J = np.zeros((n, n))  # Assuming a square Jacobian matrix for simplicity

    for i in range(n):
        j = Fj[i]
        
        if Fj[i + 1] - Fj[i] == 0:
            J[i, j] = 1
        
        elif Fj[i + 1] - Fj[i] == 1:
            J[i, j] = -1
            J[i, j + 1] = 1
        
        else:
            k = 0
            sum_val = 0
            temp = -pow(u[i], p + 1)
            
            while Fj[i + 1] - Fj[i] - k > 0:
                fig = Fi[j + k]
                J[i, j + k] = temp * pow(t[i] - t[fig] - dmin[i], -p - 1)
                sum_val += J[i, j + k]
                k += 1
            
            J[i, j + k] = -sum_val
    
    return J

'''# Example usage:
n = 5  # Number of gates/nodes
F = np.array([0, 1, 2, 3, 4])  # Example dependency structure
Fi = np.array([1, 2, 3, 4])  # Row indices
Fj = np.array([0, 1, 2, 3, 4, 4])  # Column index pointers
dmin = np.array([0.5, 0.3, 0.7, 0.2, 0.4])  # Minimum delays
u = np.array([1.5, 2.1, 1.8, 2.3, 1.7])  # Computed delays for each node
t = np.array([1.5, 2.1, 1.8, 2.3, 1.7])  # Timing values for each gate/node
p = 2.0  # Softmin parameter

Ju_t = jacobianut(n, F, Fi, Fj, dmin, u, t, p)
print("Jacobian matrix J:", Ju_t)'''

