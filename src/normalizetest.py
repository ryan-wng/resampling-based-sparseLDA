import numpy as np
from normalize import normalize

def normalizetest(Xtst, Xn):
    ntst, p = Xtst.shape
    
    # Subtract training mean from test data
    Xtst_centered = Xtst - Xn['mx']  # broadcast subtraction
    
    # Normalize only columns where norm != 0
    Xtst_normalized = Xtst_centered[:, Xn['Id']] / Xn['vx'][Xn['Id']]
    
    return Xtst_normalized