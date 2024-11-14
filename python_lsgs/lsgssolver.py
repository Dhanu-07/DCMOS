import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
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
def lsgssolver(n, a, g, F, Ft, dmin, T, A, PO, Ain, ntld, Atld, Atldsq, t0, quiet, MAXCUMPCGITERS):

    # smooth approximation parameters
    SMAXWT = 4  # soft-max weight
    SMINWT = 50  # soft-min weight

    # backtracking line search parameters
    ALPHA = 0.01  # (0, 0.5]
    BETA = 0.5  # (0, 1)
    MAXLSITER = 25
    lsiter = 0

    MAXITERS = 500
    MAXPCGITERS = 2
    TOLPCG = 1e-4

    t = t0
    x = np.zeros(n)
    u = np.zeros(n)
    dttld = np.zeros(ntld)
    umax = np.zeros(n)
    dt = np.zeros(n)
    xp = np.zeros(n)
    up = np.zeros(n)
    xhard = np.zeros(n)
    uhard = np.zeros(n)
    Ju_x = 2 * Ft + csr_matrix((-10 * np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
    Ju_t = 2 * F + csr_matrix((-10 * np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))

    gatedelays_softmin(F, dmin, SMINWT, t, u)
    nlbs_softmax(Ft, g, SMAXWT, u, x, umax)

    area = []
    areahard = []
    cumpcgiters = []
    sumpcgiters = 0
    if not quiet:
        print(f"  # \t Area(soft) \t  Area \t   CumPCGiters")

    for i in range(MAXITERS):
        if np.any(-A.T @ t - Ain.T @ dmin < 0):
            print('ERROR: Delay below dmin')
            break

        f = np.log(np.dot(a.T, x))
        gradf_x = a / (np.dot(a.T, x))
        jacobianux(Ft, x, u, umax, SMAXWT, Ju_x)
        gradf_u = np.linalg.solve(Ju_x, gradf_x)
        jacobianut(F, dmin, u, t, SMINWT, Ju_t)
        gradf_t = Ju_t @ gradf_u
        gradf_ttld = gradf_t
        gradf_t[PO] = 0
        gradf_ttld[PO] = []

        delay = -A.T @ t - Ain.T @ dmin
        H_beta = (Ain.T @ (2 * a)) / (delay ** 3)

        Hfn = lambda H_w: Atld @ ((Atld.T @ H_w) * H_beta)
        Mfn = lambda H_w: H_w / (Atldsq @ (H_beta))

        dttld, flag = cg(Hfn, -gradf_ttld, tol=TOLPCG, maxiter=MAXPCGITERS, M=Mfn, x0=dttld)
        if np.dot(gradf_ttld.T, dttld) < 0:
            dttldtodt(dttld, PO, dt)
            ddelay = -A.T @ dt

            # Backtracking
            negindex = np.where(ddelay < 0)[0]
            if len(negindex) > 0:
                s = 0.9 * np.min(-(delay[negindex] / ddelay[negindex]))
            else:
                s = 1

            if lsiter > 2:
                s *= BETA ** (lsiter - 2)

            for lsiter in range(max(lsiter - 2, 0), MAXLSITER):
                tp = t + s * dt
                gatedelays_softmin(F, dmin, SMINWT, tp, up)
                nlbs_softmax(Ft, g, SMAXWT, up, xp, umax)
                fp = np.log(np.dot(a.T, xp))
                if fp - f < ALPHA * s * np.dot(gradf_t.T, dt):
                    break
                s *= BETA

            if lsiter < MAXLSITER:
                t = tp
                u = up
                x = xp
                MAXPCGITERS = 2
                gatedelays(F, dmin, t, uhard)
                nlbs(Ft, g, uhard, xhard)

        else:
            dttld.fill(0)
            lsiter = 0
            MAXPCGITERS = 4

        if i > 2:
            if 0.05 * (area[i - 2] - area[i - 1]) > area[i - 1] - np.dot(a.T, x):
                dttld.fill(0)
                lsiter = 0
                MAXPCGITERS = 4

        area.append(np.dot(a.T, x))
        areahard.append(np.dot(a.T, xhard))

        if flag == 0:
            sumpcgiters += len(dttld)
        else:
            sumpcgiters += MAXPCGITERS
        cumpcgiters.append(sumpcgiters)

        if not quiet:
            print(f'{i:3d} {area[i]:15.3e} {areahard[i]:15.3e} {sumpcgiters:10d}')

        if sumpcgiters >= MAXCUMPCGITERS:
            break

    return t, uhard, xhard, cumpcgiters, area, areahard

