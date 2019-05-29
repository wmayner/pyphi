import numpy as np
import pyphi
import scipy.io
from datetime import datetime
import pickle

import relFunc as rel
import gridmaker as grid
from pyphi import Direction

pyphi.config.PARTITION_TYPE = 'TRI'
pyphi.config.MEASURE = 'BLD'

with open('PQR_CES_4.0.pickle', 'rb') as f:
    X = pickle.load(f)

oldconst = X.concepts
subsystem = X.subsystem
subsystem.clear_caches()

Ncon = len(oldconst)

#reconstruct the concept list so that they will have the hanging _partlist item
const = []
for x in range(0,Ncon):
    const.append(subsystem.concept(oldconst[x].mechanism))

rps = rel.getAllRelations(subsystem,const,dispoutput=True)

rps

#phis = []
#X = []
#for x in range(0,len(const)):
#    phis.append(const[x].phi)
#    X.append([const[x].cause.purview,
#          const[x].cause.mechanism,
#          const[x].effect.purview])

#fname = 'PQRsummary.mat'
#scipy.io.savemat(fname,mdict={'X':X,'phis':phis,'rps':rps})
