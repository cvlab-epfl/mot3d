import numpy as np
import cv2
from . import distance_functions as df
from .types import Detection2D, Detection3D, DetectionTracklet2D, DetectionTracklet3D

def euclidean(p1, p2):
    return np.sqrt(sum([(i-j)**2 for i,j in zip(p1,p2)])) 

def weight_distance_detections_2d(d1, d2,
                                  sigma_jump=1, sigma_distance=2,
                                  sigma_color_histogram=0.3, sigma_box_size=0.3,
                                  max_distance=20,
                                  use_color_histogram=True, use_bbox=True):
    weights = []

    dist = euclidean(d1.position, d2.position)
    if dist>max_distance:
        return None
    weights.append( np.exp(-dist**2/sigma_distance**2) )    
    
    if use_color_histogram:
        chs = 1-df.color_histogram_similarity(d1.color_histogram, d2.color_histogram)
        weights.append( np.exp(-chs**2/sigma_color_histogram**2) )
    
    if use_bbox:
        bss = 1-df.bbox_size_similarity(d1.bbox, d2.bbox)
        weights.append( np.exp(-bss**2/sigma_box_size**2) )
    
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
    
    dist = euclidean(d1.position, d2.position)
    if dist>max_distance:
        return None    
    weights.append( np.exp(-dist**2/sigma_distance**2) )    

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
        
    jump = d1.diff_index(d2)
    return -np.exp(-(jump-1)*sigma_jump) * np.prod(weights)
    #return -np.exp(-(jump-1)*a) * np.exp(-distance**2/sigma_distance**2)    

def weight_confidence_detections_3d(detection, **kwargs):
    return weight_confidence_detections(detection, **kwargs)

def weight_distance_tracklets_2d(t1, t2,
                                 sigma_color_histogram=0.3, sigma_motion=50, alpha=0.7,
                                 cutoff_motion=0.1, cutoff_appearance=0.1,
                                 max_distance=None,
                                 use_color_histogram=True, debug=False):
    
    def log(dev=-1, chs=-1, wm=-1, wa=-1, msg=''):
        if debug:
            print("head:{}:{} tail:{}:{} |dev:{:0.3f} w:{:0.3f}|color:{:0.3f} w:{:0.3f}| {}".format(t1.head.index, tuple(t1.head.position), 
                                                                                                    t2.tail.index, tuple(t2.tail.position),
                                                                                                    dev,wm, chs,wa, msg))
    if max_distance is not None:
        dist = euclidean(t1.head.position, t2.tail.position)
        if dist>max_distance:
            return None             

    # if an object is leaving the scene while another is entering the scene
    # it is possible that the two trajectories will be connected. 
    # To prevent this we simply avoid creating edges/links on the access points.
    if t2.tail.access_point:
        return None
    
    # motion model
    deviation = df.linear_motion(t1.head.indexes, t1.head.positions, 
                                 t2.tail.indexes, t2.tail.positions)    
    wm = np.exp(-deviation**2/sigma_motion**2)
    if wm<cutoff_motion:
        log(dev=deviation, wm=wm, msg='discarded: cutoff motion')
        return None    
    
    if use_color_histogram and t1.head.color_histogram is not None and t2.tail.color_histogram is not None:
        # apperance model
        chs = 1-df.color_histogram_similarity(t1.head.color_histogram, 
                                              t2.tail.color_histogram)
        wa = np.exp(-chs**2/sigma_color_histogram**2) 
        if wa<cutoff_appearance:
            log(dev=deviation, wm=wm, chs=chs, wa=wa, msg='discarded: cutoff appearance')
            return None   
        
        log(dev=deviation, wm=wm, chs=chs, wa=wa)
        return -((1-alpha)*wm + alpha*wa) # TODO: try the more complex one          
    else:
        log(dev=deviation, wm=wm, msg='weight without color hist.')
        return -wm

def weight_confidence_tracklets_2d(tracklet, **kwargs):
    return weight_confidence_detections(tracklet, **kwargs)

def weight_distance_tracklets_3d(t1, t2,
                                  sigma_color_histogram=0.3, sigma_motion=5, alpha=0.7,
                                  cutoff_motion=0.1, cutoff_appearance=0.1,
                                  max_distance=None,
                                  use_color_histogram=True):
    
    if max_distance is not None:
        dist = euclidean(t1.head.position, t2.tail.position)
        if dist>max_distance:
            return None      
    
    # motion model
    deviation = df.linear_motion(t1.head.indexes, t1.head.positions, 
                                 t2.tail.indexes, t2.tail.positions)   
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



def project_points(points, calib):
    R = np.float32(calib['R'])
    rvec = cv2.Rodrigues(R)[0]
    tvec = np.float32(calib['t'])
    K = np.float32(calib['K'])
    dist = np.float32(calib['dist'])
    
    points = np.reshape(points, (-1,3), np.float32)
    return cv2.projectPoints(points, rvec, tvec, K, dist)[0].reshape(-1,2)

def weight_distance_tracklet3d_tracklet2d(t1, t2, calib={}, **kwargs):
    # t1: 3D tracklet t2: 2D tracklet
    
    view = t2.view
    
    proj = project_points(t1.head.positions, calib[view])
    indexes = t1.head.indexes
 
    t1_2d = [Detection2D(index, position, view=view) 
               for index,position in zip(indexes, proj)]
    
    return weight_distance_tracklets_2d(DetectionTracklet2D(t1_2d), t2, **kwargs)
    
def weight_distance_tracklet2d_tracklet3d(t1, t2, calib={}, **kwargs):
    # t1: 2D tracklet t2: 3D tracklet
    
    view = t1.view
    
    proj = project_points(t2.tail.positions, calib[view])
    indexes = t2.tail.indexes
 
    t2_2d = [Detection2D(index, position, view=view) 
               for index,position in zip(indexes, proj)]
    
    return weight_distance_tracklets_2d(t1, DetectionTracklet2D(t2_2d), **kwargs)
    
def weight_distance_tracklets_auto(t1, t2, config2d={}, config3d={}, calib={}):

    is_t1_2D = isinstance(t1, DetectionTracklet2D) 
    is_t2_2D = isinstance(t2, DetectionTracklet2D)    
    is_t1_3D = isinstance(t1, DetectionTracklet3D)
    is_t2_3D = isinstance(t2, DetectionTracklet3D)
    
    if is_t1_2D and is_t2_2D:
        return weight_distance_tracklets_2d(t1, t2, **config2d)
    elif is_t1_3D and is_t2_3D:
        return weight_distance_tracklets_3d(t1, t2, **config3d)
    
    elif is_t1_3D and is_t2_2D:
        return weight_distance_tracklet3d_tracklet2d(t1, t2, calib=calib, **config2d)
    
    elif is_t1_2D and is_t2_3D:
        return weight_distance_tracklet2d_tracklet3d(t1, t2, calib=calib, **config2d) 
    
    else:
        raise RuntimeError("Unable to pick the pick the correct weight distance for tracklets [{},{}].".format(type(t1), type(t2)))
        
def weight_confidence_tracklets_auto(t, config2d={}, config3d={}):

    is_t_2D = isinstance(t, DetectionTracklet2D) 
    is_t_3D = isinstance(t, DetectionTracklet3D)
    
    if is_t_2D:
        return weight_confidence_tracklets_2d(t, **config2d)
    elif is_t_3D:  
        return weight_confidence_tracklets_3d(t, **config3d)
    else:
        raise RuntimeError("Unable to pick the pick the correct weight confidence for tracklets [{}].".format(type(t)))     
        
def weight_distance_detections_auto(d1, d2, config2d={}, config3d={}):

    is_d1_2D = isinstance(d1, Detection2D) 
    is_d2_2D = isinstance(d2, Detection2D)    
    is_d1_3D = isinstance(d1, Detection3D)
    is_d2_3D = isinstance(d2, Detection3D)
    
    if is_d1_2D and is_d2_2D:
        return weight_distance_detections_2d(d1, d2, **config2d)
    elif is_d1_3D and is_d2_3D:
        return weight_distance_detections_3d(d1, d2, **config3d)
    
    elif is_d1_3D and is_d2_2D:
        raise RuntimeError("Combination not possible or not implemented [{},{}].".format(type(d1), type(d2)))  
    
    elif is_d1_2D and is_d2_3D:
        raise RuntimeError("Combination not possible or not implemented [{},{}].".format(type(d1), type(d2)))   
    
    else:
        raise RuntimeError("Unable to pick the pick the correct weight distance for detections [{}].".format(type(d1), type(d2)))       
        
def weight_confidence_detections_auto(d, config2d={}, config3d={}):

    is_d_2D = isinstance(d, Detection2D)
    is_d_3D = isinstance(d, Detection3D)
    
    if is_d_2D:
        return weight_confidence_detections_3d(d, **config2d)
    elif is_d_3D:
        return weight_confidence_detections_3d(d, **config3d)
    else:
        raise RuntimeError("Unable to pick the pick the correct weight confidence for detections [{}].".format(type(d)))     
       