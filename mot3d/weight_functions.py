import numpy as np
import cv2
from . import distance_functions as df

def euclidean(p1, p2):
    return np.sqrt(sum([(i-j)**2 for i,j in zip(p1,p2)])) 

def weight_distance_detections_2d(d1, d2,
                                  sigma_jump=1, sigma_distance=2,
                                  sigma_color_histogram=0.3, sigma_box_size=0.3,
                                  max_distance=20,
                                  use_color_histogram=True, use_bbox=True):
    weights = []

    if use_color_histogram:
        chs = 1-df.color_histogram_similarity(d1.color_histogram, d2.color_histogram)
        weights.append( np.exp(-chs**2/sigma_color_histogram**2) )
    
    if use_bbox:
        bss = 1-df.bbox_size_similarity(d1.bbox, d2.bbox)
        weights.append( np.exp(-bss**2/sigma_box_size**2) )
    
    dist = euclidean(d1.position, d2.position)
    if dist>max_distance:
        return None
    weights.append( np.exp(-dist**2/sigma_distance**2) )
    
    jump = d1.diff_index(d2)
    return -np.exp(-(jump-1)*sigma_jump) * np.prod(weights)
    #return -np.exp(-(jump-1)*a) * np.exp(-distance**2/sigma_distance**2)
    
def weight_confidence_detections(detection, mul=1, bias=0):
    return -(detection.confidence*mul + bias) 

def weight_confidence_detections_2d(detection, **kwargs):
    return weight_confidence_detections(detection, **kwargs)
    
def weight_distance_detections_3d(d1, d2,
                                  sigma_jump=1, sigma_distance=2,
                                  sigma_color_histogram=0.3, sigma_box_size=0.3,
                                  max_distance=5,
                                  use_color_histogram=True, use_bbox=True):
    weights = []

    if use_color_histogram:
        _similarities = []
        for view in d1.detections_2d.keys():
            if view in d2.detections_2d:
                chs = 1-df.color_histogram_similarity(d1.detections_2d[view].color_histogram, 
                                                      d2.detections_2d[view].color_histogram)
                _similarities.append(chs)
        weights.append( np.exp(-np.mean(_similarities)**2/sigma_color_histogram**2) )
    
    if use_bbox:
        _similarities = []
        for view in d1.detections_2d.keys():
            if view in d2.detections_2d:
                bss = 1-df.bbox_size_similarity(d1.detections_2d[view].bbox, 
                                                d2.detections_2d[view].bbox)
                _similarities.append(bss)
        weights.append( np.exp(-np.mean(_similarities)**2/sigma_box_size**2) )
    
    dist = euclidean(d1.position, d2.position)
    if dist>max_distance:
        return None    
    weights.append( np.exp(-dist**2/sigma_distance**2) )
    
    jump = d1.diff_index(d2)
    return -np.exp(-(jump-1)*sigma_jump) * np.prod(weights)
    #return -np.exp(-(jump-1)*a) * np.exp(-distance**2/sigma_distance**2)    

def weight_confidence_detections_3d(detection, **kwargs):
    return weight_confidence_detections(detection, **kwargs)

def weight_distance_trackelts_2d(t1, t2, 
                                 sigma_color_histogram=0.3, sigma_motion=50, alpha=0.7,
                                 cutoff_motion=0.1, cutoff_appearance=0.1,
                                 use_color_histogram=True):
    
    # if an object is leaving the scene while another is entering the scene
    # it is possible that the two trajectories will be connected. 
    # To prevent this we simply avoid creating edges/links on the access points.
    if t2.tail.access_point:
        return None
    
    # motion model
    deviation = df.linear_motion(t1.head.indexes, t1.head.positions, 
                                 t2.tail.indexes, t2.tail.positions, sigma_motion)    
    wm = np.exp(-deviation**2/sigma_motion**2)
    if wm<cutoff_motion:
        return None    
    
    if use_color_histogram:
        # apperance model
        chs = 1-df.color_histogram_similarity(t1.head.color_histogram, 
                                              t2.tail.color_histogram)
        wa = np.exp(-chs**2/sigma_color_histogram**2) 
        if wa<cutoff_appearance:
            return None   
        
        return -((1-alpha)*wm + alpha*wa) # TODO: try the more complex one          
    else:
        return -wm

def weight_confidence_tracklets_2d(tracklet, **kwargs):
    return weight_confidence_detections(tracklet, **kwargs)

def weight_distance_trackelts_3d(t1, t2, 
                                  sigma_color_histogram=0.3, sigma_motion=5, alpha=0.7,
                                  cutoff_motion=0.1, cutoff_appearance=0.1,
                                  use_color_histogram=True):
    
    # motion model
    deviation = df.linear_motion(t1.head.indexes, t1.head.positions, 
                                 t2.tail.indexes, t2.tail.positions, 
                                 sigma_motion)   
    wm = np.exp(-deviation**2/sigma_motion**2)
    if wm<cutoff_motion:
        return None    
    
    if use_color_histogram:
        # apperance model
        _similarities = []
        for view in t1.tracklets_2d.keys():
            if view in t2.tracklets_2d:
                chs = 1-df.color_histogram_similarity(t1.tracklets_2d[view].head.color_histogram, 
                                                      t2.tracklets_2d[view].tail.color_histogram)
                _similarities.append(chs)
        wa = np.exp(-np.mean(_similarities)**2/sigma_color_histogram**2) 
        if wa<cutoff_appearance:
            return None  
    
        return -((1-alpha)*wm + alpha*wa) # TODO: try the more complex one
    else:
        return -wm

def weight_confidence_tracklets_3d(tracklet, **kwargs):
    return weight_confidence_detections(tracklet, **kwargs)