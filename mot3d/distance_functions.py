#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# ---------------------------------------------------------------------------
import cv2
import numpy as np

__all__ = ["color_histogram", "color_histogram_similarity", "bbox_size_similarity", "linear_motion"]

def color_histogram(patch, channels=(0,1,2), N=255):
    
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    
    H = []
    for i in channels:
        h,_ = np.histogram(patch_lab[:,:,i].ravel(), bins=N, range=(0,255))
        h = h-h.mean()
        H.append(h)
    return H

def color_histogram_similarity(H1, H2, weights=None):
    
    n_channels = len(H1)
    
    if weights is None:
        weights = tuple(1 for _ in range(n_channels))
           
    scs = 0
    for i in range(n_channels):
        
        sc = np.sum(H1[i]*H2[i])/np.sqrt(np.sum(H1[i]**2)*np.sum(H2[i]**2))
        scs += weights[i]*sc
        
    return scs/n_channels

def bbox_size_similarity(bbox1, bbox2):
    
    # bbox = (xmin, ymin, xmax, ymax)
    size = lambda bbox: (bbox[3]-bbox[1],bbox[2]-bbox[0])
    
    h1,w1 = size(bbox1)
    h2,w2 = size(bbox2)
           
    return 1-(np.abs(h1-h2)/(np.maximum(h1,h2)+1e-12)+\
              np.abs(w1-w2)/(np.maximum(w1,w2)+1e-12))/2

def linear_motion(idxs1, pts1, idxs2, pts2):
    
    assert idxs2[0]>idxs1[-1], "idxs2[0]={} idxs1[-1]={} tracklet2 cannot be more recent than tracklet1!".format(idxs2[0],idxs1[-1])
    
    def linear_deviation(pts1, idxs1, pts2, idxs2):
        try:
            polys = np.polyfit(idxs1, pts1, 1)
        except:
            print("np.polyfit error: idxs1={}, pts1={}".format(idxs1, pts1))
            raise
        pts2p = np.vstack([np.polyval(p, idxs2) for p in polys.T]).T
        avg_dist = np.linalg.norm(pts2p-pts2, axis=1).mean()
        return avg_dist    

    forward_deviation = linear_deviation(pts1, idxs1, pts2, idxs2)
    backward_deviation = linear_deviation(pts2, idxs2, pts1, idxs1)       

    return forward_deviation+backward_deviation