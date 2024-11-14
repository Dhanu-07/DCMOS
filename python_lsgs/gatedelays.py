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
def gatedelays(n, F, Fi, Fj, dmin, t):
    u = np.zeros(n)

    for i in range(n):
        u[i] = t[i] - dmin[i]
        k = 0
        while Fj[i + 1] - Fj[i] - k > 0:
            j = Fj[i] + k
            fig = Fi[j]
            u[i] = min(u[i], t[i] - t[fig] - dmin[i])
            k += 1

    return u

'''# Example usage:
#n = 5  # Number of gates/nodes
#F = np.array([0, 1, 2, 3, 4])  # Example values for F (dependency weights if needed)
#Fi = np.array([1, 2, 3, 4])  # Example connections (row indices)
Fj = np.array([0, 1, 2, 3, 4, 4])  # Column index pointers
dmin = np.array([0.5, 0.3, 0.7, 0.2, 0.4])  # Minimum delays
t = np.array([1.5, 2.1, 1.8, 2.3, 1.7])  # Timing values for each gate/node

u = gatedelays(n, F, Fi, Fj, dmin, t)
print("Resulting u:", u)'''

