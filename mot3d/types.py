import numpy as np
from collections import defaultdict

__all__ = ["Detection", "Detection2D", "Detection3D", 
           "DetectionTracklet", "DetectionTracklet2D", "DetectionTracklet3D"] 

class Detection(object):

    def __init__(self, index, confidence=0.5, id=None):
        self.index = index
        self.confidence = confidence
        self.id = id        
        
        self.pre_node = None
        self.post_node = None
        
        self.weight_source = None
        self.weight_sink = None
        
        self.entry_points = True
        self.exit_point = True
        
    def diff_index(self, other_detection):
        return other_detection.index-self.index
        
class Detection2D(Detection):
    
    def __init__(self, index, position, confidence=0.5, id=None, 
                 view=None, color_histogram=None, bbox=None):    
        super(Detection2D, self).__init__(index, confidence, id)
        self.position = position
        self.view = view    
        self.color_histogram = color_histogram
        self.bbox = bbox

class Detection3D(Detection):
    
    def __init__(self, index, position, confidence=0.5, id=None, detections_2d={}):    
        super(Detection3D, self).__init__(index, confidence, id)
        self.position = position
        self.detections_2d = detections_2d

class DetectionTracklet(object):
    
    def __init__(self, tracklet, confidence=0.5, id=None):
        
        self.tracklet = tracklet
        self.confidence = confidence
        self.id = id
        
        self.head = tracklet[-1]
        self.tail = tracklet[0]  
        
        self.pre_node = None
        self.post_node = None
        
        self.weight_source = None
        self.weight_sink = None
        
        self.entry_points = True
        self.exit_point = True        
        
    def diff_index(self, other_tracklet):
        return other_tracklet.tail.index-self.head.index  
    
def _extract_features(tracklet):
    
    indexes, positions, color_histograms, bboxes = [],[],[],[]
    for detection in tracklet:
        positions.append(detection.position)
        indexes.append(detection.index)    
        if detection.color_histogram is not None:
            color_histograms.append(detection.color_histogram)
        if detection.bbox is not None:
            bboxes.append(detection.bbox)

    color_histogram, bbox = None, None
    if len(color_histograms):
        color_histogram = np.median(color_histograms, axis=0)
    if len(bboxes):
        bbox = np.median(bboxes, axis=0)    
        
    return indexes, positions, color_histogram, bbox

def _check(tracklet, type):
    for d in tracklet:
        if not isinstance(d, type):
            raise ValueError("The tracklet must be composed of objects of type {} only! ({})".format(type.__name__, type(d)))           

class DetectionTracklet2D(DetectionTracklet):
    
    def __init__(self, tracklet, confidence=0.5, window=None, view=None):
        super(DetectionTracklet2D, self).__init__(tracklet, confidence, id)
        
        _check(tracklet, Detection2D)
        self.confidence = confidence
        self.window = window
        self.view = view

        # head: the part of the trajectory that is the most recent in time
        if window is not None:
            cutoff = tracklet[-1].index-window
            tracklet_head = [d for d in tracklet if d.index>cutoff]
        else:
            tracklet_head = tracklet
        indexes, positions, color_histogram, bbox = _extract_features(tracklet_head)        
        
        self.head = Detection2D(indexes[-1], positions[-1], view=view, 
                                color_histogram=color_histogram, bbox=bbox)
        self.head.indexes = indexes
        self.head.positions = positions
        self.head.access_point = False
        
        # tail: the part of the trajectory that is the oldest in time
        if window is not None:
            cutoff = tracklet[0].index+window
            indexes, positions, color_histogram, bbox = _extract_features([d for d in tracklet if d.index<cutoff])   
        else:
            # use the same we computed before
            pass
        
        self.tail = Detection2D(indexes[0], positions[0], view=view, 
                                color_histogram=color_histogram, bbox=bbox)
        self.tail.indexes = indexes
        self.tail.positions = positions  
        self.tail.access_point = False
        
def _extract_features_3d(tracklet):
    
    indexes, positions = [],[]
    for detection in tracklet:
        positions.append(detection.position)
        indexes.append(detection.index)
        
    return indexes, positions
        
class DetectionTracklet3D(DetectionTracklet):
    
    def __init__(self, tracklet, confidence=0.5, window=None):
        super(DetectionTracklet3D, self).__init__(tracklet, confidence, id)
        
        _check(tracklet, Detection3D)
        self.confidence = confidence     
        self.window = window      
        
        self.tracklets_2d = defaultdict(lambda: [])
        for detection in tracklet:
            for view, detection_2d in detection.detections_2d.items():
                self.tracklets_2d[view].append(detection_2d)
                
        self.tracklets_2d = {view: DetectionTracklet2D(tracklet, view=view, window=window) 
                             for view,tracklet in self.tracklets_2d.items()}
        
        # head: the part of the trajectory that is the most recent in time
        if window is not None:
            cutoff = tracklet[-1].index-window       
            tracklet_head = [d for d in tracklet if d.index>cutoff]
        else:
            tracklet_head = tracklet
        indexes, positions = _extract_features_3d(tracklet_head)         
        
        self.head = Detection3D(indexes[-1], positions[-1])
        self.head.indexes = indexes
        self.head.positions = positions
        self.head.access_point = False
            
        # tail: the part of the trajectory that is the oldest in time
        if window is not None:
            cutoff = tracklet[0].index+window
            indexes, positions = _extract_features_3d([d for d in tracklet if d.index<cutoff])
        else:
            # use the same we computed before
            pass
        
        self.tail = Detection3D(indexes[0], positions[0])
        self.tail.indexes = indexes
        self.tail.positions = positions 
        self.tail.access_point = False