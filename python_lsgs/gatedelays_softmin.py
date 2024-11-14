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
def gatedelays_softmin(n, F, Fi, Fj, dmin, p, t):
    u = np.zeros(n)

    for i in range(n):
        k = 0
        minui = t[i] - dmin[i]
        
        while Fj[i + 1] - Fj[i] - k > 0:
            j = Fj[i] + k
            fig = Fi[j]
            minui = min(t[i] - t[fig] - dmin[i], minui)
            k += 1
        
        if k <= 1:
            u[i] = minui
        else:
            u[i] = 0
            k = 0
            while Fj[i + 1] - Fj[i] - k > 0:
                j = Fj[i] + k
                fig = Fi[j]
                u[i] += (minui / (t[i] - t[fig] - dmin[i])) ** p
                k += 1
            u[i] = minui * (u[i] ** (-1 / p))

    return u

'''# Example usage:
n = 5  # Number of gates/nodes
F = np.array([0, 1, 2, 3, 4])  # Example dependency structure
Fi = np.array([1, 2, 3, 4])  # Row indices
Fj = np.array([0, 1, 2, 3, 4, 4])  # Column index pointers
dmin = np.array([0.5, 0.3, 0.7, 0.2, 0.4])  # Minimum delays
p = 2.0  # Softmin parameter
t = np.array([1.5, 2.1, 1.8, 2.3, 1.7])  # Timing values for each gate/node

u = gatedelays_softmin(n, F, Fi, Fj, dmin, p, t)
print("Resulting u:", u)'''

