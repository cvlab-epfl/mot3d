#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# ---------------------------------------------------------------------------
import numpy as np

def hungarian_matching(pos1, pos2, radius_match=0.3):
    from munkres import Munkres
    
    pos1 = np.reshape(pos1, (-1,2))
    pos2 = np.reshape(pos2, (-1,2))   

    n1 = pos1.shape[0]    
    n2 = pos2.shape[0]

    n_max = max(n1, n2)
    
    if n_max==0:
        return None, None, [], [], []

    # building the cost matrix based on the distance between 
    # detections and ground-truth positions
    matrix = np.ones((n_max, n_max))*9999999
    for i in range(n1):    
        for j in range(n2):

            d = np.sqrt(((pos2[j,0] - pos1[i,0])**2 + \
                         (pos2[j,1] - pos1[i,1])**2))

            if d <= radius_match:
                matrix[i,j] = d

    indexes = Munkres().compute(matrix.copy())

    TP = []   
    matched1 = np.zeros(len(pos1), np.bool)
    matched2 = np.zeros(len(pos2), np.bool)
    for i, j in indexes:
        value = matrix[i][j]
        if value <= radius_match:
            TP.append(j)
            matched1[i] = True
            matched2[j] = True
        else:
            TP.append(None)
            
    FP = np.where(matched2==False)[0].tolist()
    FN = np.where(matched1==False)[0].tolist()

    return TP, FP, FN