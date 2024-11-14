import numpy as np
from scipy.sparse import lil_matrix
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
from lsgssolver import lsgssolver
from nlbs_softmax import *
from nlbs import *
from timingasg import *

# Number of gates
n = 6

# Delay constraints
dmin = np.array([2.3310, 0.9990, 0.9990, 0.9990, 1.9980, 1.9980])

# Gate parameters (logical effort)
g = np.array([1.4005, 1.3484, 1.0100, 1.9427, 8.3406, 9.1768])

# Circuit connection matrix (using sparse matrix)
F = lil_matrix((n, n))
F[1, 2] = 0.9990
F[1, 3] = 0.9990
F[0, 4] = 1.3320
F[2, 4] = 1.3320
F[2, 5] = 1.6650
F[3, 5] = 1.6650

# Gate areas
a = np.array([16, 3, 3, 3, 8, 10])

# Timing specification
T = 12
print(f"\nSizing ISCAS85 c17 circuit for timing specification T = {T:.2f}")
x = lsgssolver(n, a, g, F, F.T, dmin, T, None, None, 250, None, None, None, None, None, 250)
print('Optimal gate sizes are:')
print(x)

# Optimize ISCAS85 c17 circuit with all initial gate sizes = 2
x0 = 2 * np.ones(n)
print('\nOptimizing ISCAS85 c17 circuit with all initial gate sizes = 2')
x, t, d, cumpcgiters, areahard, areasoft = lsgssolver(n, a, g, F, F.T, dmin, None, x0, None, 250, None, None, None, None, None, 250)

# Calculate area reduction ratio
original_area = np.dot(a.T, x0)
optimal_area = np.dot(a.T, x)
area_reduction_ratio = (original_area - optimal_area) / original_area

# Display results
print(f'\nOriginal area (all gates sizes = 2): {original_area:.3f}')
print(f'Optimal area for the same circuit delay: {optimal_area:.3f}')
print(f'Area reduction ratio ((original-optimal)/original): {area_reduction_ratio:.3f}')
print('Optimal gate sizes are:')
print(x)

