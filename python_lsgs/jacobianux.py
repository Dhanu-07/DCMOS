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
def jacobianux(n, Ft, Fti, Ftj, x, u, umax, p):
    J = np.zeros((n, n))  # Assuming a square Jacobian matrix for simplicity

    for i in range(n):
        j = Ftj[i]
        J[i, j] = -u[i] / (x[i] * (1 - pow(x[i], -p)))

        k = 0
        while Ftj[i + 1] - Ftj[i] - k > 0:
            j = Ftj[i] + k
            J[i, j + 1] = u[i] * (Ft[j] / umax[i])
            k += 1

    return J

'''# Example usage:
n = 5  # Number of gates/nodes
Ft = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Example values for Ft
Fti = np.array([1, 2, 3, 4])  # Row indices
Ftj = np.array([0, 1, 2, 3, 4, 4])  # Column index pointers
x = np.array([1.5, 2.1, 1.8, 2.3, 1.7])  # Gate sizing variables
u = np.array([1.5, 2.1, 1.8, 2.3, 1.7])  # Computed delays for each node
umax = np.array([1.6, 2.0, 1.9, 2.2, 1.8])  # Maximum delays for each node
p = 2.0  # Softmin parameter

Ju_x = jacobianux(n, Ft, Fti, Ftj, x, u, umax, p)
print("Jacobian matrix J:", Ju_x)'''

