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
def nlbs(n, Ft, Fti, Ftj, g, u, x):
    for i in range(n - 1, -1, -1):  # Process nodes in reverse order
        t = g[i]
        k = 0
        while Ftj[i + 1] - Ftj[i] - k > 0:
            j = Ftj[i] + k
            fog = Fti[j]
            t += Ft[j] * x[fog]
            k += 1
        x[i] = max(1, t / u[i])

    return x

'''# Example usage:
n = 5  # Number of gates/nodes
Ft = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Example values for Ft
Fti = np.array([1, 2, 3, 4])  # Row indices
Ftj = np.array([0, 1, 2, 3, 4, 4])  # Column index pointers
g = np.array([1.2, 1.3, 1.4, 1.1, 1.5])  # Some g values for each node
u = np.array([1.5, 2.1, 1.8, 2.3, 1.7])  # Computed delays for each node
x = np.ones(n)  # Initialize x with ones

x = nlbs(n, Ft, Fti, Ftj, g, u, x)
print("Updated x values:", x)'''

