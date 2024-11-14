import numpy as np
import scipy.sparse as sp
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
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio


mat_data = sio.loadmat('/home/arunp24/python_lsgs/examples/ckt10k.mat')

# Assuming 'a', 'dmin', 'g', and 'F' are the variable names in the .mat file
a = mat_data['a'].flatten()  # Flatten to 1D if it's a row/column vector
dmin = mat_data['dmin'].flatten()  # Flatten to 1D
g = mat_data['g'].flatten()  # Flatten to 1D

# Assuming 'F' is stored as a sparse matrix in MATLAB format (e.g., sparse COO or CSC)
F = sp.csr_matrix(mat_data['F'])  # Convert to CSR sparse matrix
T = 120
print(f"\nSizing ISCAS85 circuit for timing specification T = {T:.2f}")
def lsgs(a, g, F, dmin, T, x0=None, quiet=True, MAXCUMPCGITERS=500):
    if x0 is None:
        x0 = np.ones(len(a))

    # Input validation
    n = F.shape[0]
    if not sp.issparse(F) or F.shape[0] != F.shape[1]:
        raise ValueError("F should be a sparse strictly upper triangular matrix of size n x n.")
    
    if (a.shape[0] != n or g.shape[0] != n or dmin.shape[0] != n or F.shape[1] != n):
        raise ValueError("Size mismatch for input parameters.")
    
    if np.any(a < 0) or np.any(g < 0) or np.any(dmin < 0) or np.any(F.toarray() < 0):
        raise ValueError("Input contains negative entries.")

    if np.any(sp.tril(F).toarray()):
        raise ValueError("F should be strictly upper triangular.")

    if T is None:
        if x0 is not None:
            t0 = np.zeros(n)
            mexFunction(F, (F.dot(x0) + g) / x0 + dmin, t0)
            T = np.max(t0)
            initialize = 0
        else:
            quiet = False
            initialize = -1
    else:
        initialize = 1

    if not quiet:
        print("Input checked.")

    # Other calculations
    PO = np.where(F.sum(axis=1) == 0)[0]  # Primary Outputs
    PI = np.where(F.sum(axis=0) == 0)[0]  # Primary Inputs

    Fi, Fj = F.nonzero()
    m = Fi.size

    cumsum_PI = np.cumsum(np.ones(int(len(PI)))).astype(int) 
    cumsum_Fi = np.cumsum(np.ones(m)).astype(int) 
    cumsum_Fj = np.cumsum(np.ones(m)).astype(int) 
    A1 = sp.csr_matrix((np.ones(len(PI)), (PI, cumsum_PI))) 
    A2 = sp.csr_matrix((np.ones(m), (Fi, cumsum_Fi))) 
    A3 = sp.csr_matrix((np.ones(m), (Fj, cumsum_Fj))) # Make sure all sparse matrices have the same number of rows 
    max_rows = max(A1.shape[0], A2.shape[0], A3.shape[0]) # Pad A1, A2, and A3 to have the same number of rows (max_rows) 
    A1 = sp.vstack([A1, sp.csr_matrix((max_rows - A1.shape[0], A1.shape[1]))]) 
    A2 = sp.vstack([A2, sp.csr_matrix((max_rows - A2.shape[0], A2.shape[1]))]) 
    A3 = sp.vstack([A3, sp.csr_matrix((max_rows - A3.shape[0], A3.shape[1]))]) # Now stack the matrices horizontally A =
    
    A = sp.hstack([A1, A2,A3])
    Ain = A < 0
    Atld = A.copy()
    Atld[PO, :] = 0
    Atldsq = Atld.power(2)
    ntld = Atld.shape[0]

    Ft = F.transpose().tocsc()

    tminarea = np.zeros(n)
    mexFunction(F, dmin, tminarea)

    if not quiet:
        print(f'Circuit summary:\nNumber of gates: {n}\nNumber of interconnections: {m}')
        print(f'Circuit delay for minimum area: {np.max(tminarea):.3f}')

    lpathtoin = np.zeros(n)
    lpathtoout = np.zeros(n)
    tmin = np.zeros(n)
    tmax = np.zeros(n)

    if T is None:
        T = np.max(tminarea) - 0.01

    mexFunction_slck(F, Ft, dmin, T)

    if not quiet:
        print(f'Minimum circuit delay (Tmin): {np.max(tmin):.3f}')

    if T >= np.max(tminarea):
        print("WARNING: Requested timing specification greater than or equal to the circuit delay of the minimum area circuit.")
        x = np.ones(n)
        d = (F.sum(axis=1) + g + dmin)
        t = tminarea
        cumpcgiters = 0
        area = np.sum(a)
        areasoft = area
        return x, t, d, cumpcgiters, area, areasoft
    elif T <= np.max(tmin):
        print("ERROR: Requested timing specification is infeasible.")
        return

    if initialize == 1:
        t0 = np.zeros(n)
        lpath = lpathtoin + lpathtoout
        cslack = tmax - tmin
        u0 = cslack / lpath
        d0 = u0 + dmin
        timingasg(F, d0, t0)
    elif initialize == -1:
        return

    t0[PO] = T

    # Call the solver (lsgssolver)
    t, u, x, cumpcgiters, areasoft, area = lsgssolver(n, a, g, F, Ft, dmin, T, A, PO, Ain, ntld, Atld, Atldsq, t0, quiet, MAXCUMPCGITERS)

    d = u + dmin
    cumpcgiters = cumpcgiters.T
    areasoft = areasoft.T
    area = area.T

    return x, t, d, cumpcgiters, area, areasoft

x= lsgs(a, g, F, dmin, T, x0=None, quiet=True, MAXCUMPCGITERS=500)
print(f"Gate sizes :{x}")
# Define auxiliary functions (timingasg, cslackandlpath, lsgssolver)
# These functions are assumed to be defined elsewhere in the code or need to be implemented based on the MATLAB code's logic.

