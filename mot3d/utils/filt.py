#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# ---------------------------------------------------------------------------
import numpy as np

def merge_close_points(points, eps=0.3, return_indexes=False):
    from sklearn.cluster import DBSCAN
    
    if len(points)==0:
        return points
    
    points_ = np.array(points)
    
    clu = DBSCAN(eps=eps, min_samples=1, metric='euclidean', algorithm='brute').fit(points_)

    new_points = []
    for i in set(clu.labels_):
        if return_indexes:
            new_points.append(np.where(clu.labels_==i)[0].tolist())
        else:
            new_points.append(np.mean([points_[j] for j in np.where(clu.labels_==i)], axis=1))
    if not return_indexes:
        new_points = np.vstack(new_points)
    return new_points