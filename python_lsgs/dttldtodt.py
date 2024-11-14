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
def dttldtodt(n, dttld, PO):
    dt = np.zeros(n)
    j, k = 0, 0

    for i in range(n):
        if k < len(PO) and i == int(PO[k]) - 1:
            dt[i] = 0
            k += 1
        else:
            dt[i] = dttld[j]
            j += 1
    
    return dt



