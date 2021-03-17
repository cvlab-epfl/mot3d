import numpy as np

__all__ = ["Detection", "Detection2D", "Detection3D", "DetectionTracklet", "DetectionTracklet2D", "DetectionTracklet3D"] 

class Detection(object):
    """
    Detection class. 
    
    Inherit this class if you want to use more information other 
    then just the spatial distance to compare detections.
    """
    
    def __init__(self, index, position, confidence=0.5, datetime=None, features={}, id=None):
        self.index = index
        self.position = position
        self.confidence = confidence
        self.datetime  = datetime
        self.features = features
        self.id = id
        self.pre_node = None
        self.post_node = None
        
        self.weight_source = None
        self.weight_sink = None
        
class Detection2D(Detection):
    
    def __init__(self, index, position, confidence=0.5, datetime=None, features={}, id=None, view=None):    
        super(Detection2D, self).__init__(index, position, confidence, datetime, features, id)
        self.view = view    

class Detection3D(Detection):
    
    def __init__(self, index, position, confidence=0.5, datetime=None, features={}, id=None, detections_2d=[]):    
        super(Detection3D, self).__init__(index, position, confidence, datetime, features, id)
        self.detections_2d = detections_2d

class DetectionTracklet(object):
    
    def __init__(self, id, tracklet, window=10, confidence=0.5):
        
        self.id = id
        self.tracklet = tracklet
        self.window = window
        self.confidence = confidence
        self.pre_node = None
        self.post_node = None
        self.weight_source = None
        self.weight_sink = None         

class DetectionTracklet2D(DetectionTracklet):
    
    def __init__(self, id, tracklet, view=None, window=10, confidence=0.5):
        super(DetectionTracklet2D, self).__init__(id, tracklet, window, confidence)
        
        # check all detections are 2D ones
        for d in tracklet:
            if not isinstance(d, Detection2D):
                raise ValueError("The tracklet must be composed of objects of type Detection2D only! ({})".format(type(d)))
        
        self.view = view
        
        # --- Head of the tracklet ---
        cutoff = tracklet[-1].index-window
        color_histogram = []
        bbox = []
        positions = []
        indexes = []
        for detection in reversed(tracklet):
            if detection.index<cutoff:
                break
            positions.append(detection.position)
            indexes.append(detection.index)               
            if 'color_histogram' in detection.features:
                color_histogram.append(detection.features['color_histogram'])
            if 'bbox' in detection.features:
                bbox.append(detection.features['bbox'])

        features = {}
        if len(color_histogram):
            features['color_histograms'] = color_histogram
            features['color_histogram'] = np.mean(color_histogram, axis=0)
        if len(bbox):
            features['bboxes'] = bbox
            features['bbox'] = np.median(bbox, axis=0)
        if len(positions):
            features['positions'] = list(reversed(positions))
        if len(indexes):
            features['indexes'] = list(reversed(indexes))              

        self.head = Detection(index=tracklet[-1].index,
                              position=tracklet[-1].position,
                              features=features)        
        
        # --- Tail of the tracklet ---
        cutoff = tracklet[0].index+window
        color_histogram = []
        bbox = []
        positions = []
        indexes = []        
        for detection in tracklet:
            if detection.index>cutoff:
                break
            positions.append(detection.position)
            indexes.append(detection.index)                 
            if 'color_histogram' in detection.features:
                color_histogram.append(detection.features['color_histogram'])
            if 'bbox' in detection.features:
                bbox.append(detection.features['bbox'])

        features = {}
        if len(color_histogram):
            features['color_histograms'] = color_histogram
            features['color_histogram'] = np.mean(color_histogram, axis=0)
        if len(bbox):
            features['bboxes'] = bbox
            features['bbox'] = np.median(bbox, axis=0)
        if len(positions):
            features['positions'] = positions   
        if len(indexes):
            features['indexes'] = indexes              

        self.tail = Detection(index=tracklet[0].index,
                              position=tracklet[0].position,
                              features=features) 
        
class DetectionTracklet3D(DetectionTracklet):
    
    def __init__(self, id, tracklet, window=10, confidence=0.5):
        super(DetectionTracklet3D, self).__init__(id, tracklet, window, confidence)
        
        # check all detections are 3D ones
        for d in tracklet:
            if not isinstance(d, Detection3D):
                raise ValueError("The tracklet must be composed of objects of type Detection3D only! ({})".format(type(d)))
                
        # --- Head of the tracklet ---
        cutoff = tracklet[-1].index-window
        positions = []
        indexes = []
        for detection in reversed(tracklet):
            if detection.index<cutoff:
                break
            positions.append(detection.position)
            indexes.append(detection.index)   
            
        features = {}
        if len(positions):
            features['positions'] = list(reversed(positions))
        if len(indexes):
            features['indexes'] = list(reversed(indexes))            
        
        self.head = Detection(index=tracklet[-1].index,
                              position=tracklet[-1].position,
                              features=features)        
                     
        # --- Tail of the tracklet ---
        cutoff = tracklet[0].index+window
        positions = []
        indexes = []        
        for detection in tracklet:
            if detection.index>cutoff:
                break
            positions.append(detection.position)
            indexes.append(detection.index)                 
            
        features = {}
        if len(positions):
            features['positions'] = positions   
        if len(indexes):
            features['indexes'] = indexes             

        self.tail = Detection(index=tracklet[0].index,
                              position=tracklet[0].position,
                              features=features)                 
                
        # recover 2D tracklets
        self.tracklets_2d = {}
        for d3d in tracklet:
            for d2d in d3d.detections_2d:
                if d2d.view not in self.tracklets_2d:
                    self.tracklets_2d[d2d.view] = [d2d]
                else:
                    self.tracklets_2d[d2d.view].append(d2d)

        # prepare 2D tracklets detections as well
        for view, tracklet in self.tracklets_2d.items():
            self.tracklets_2d[view] = DetectionTracklet2D(None, tracklet, view, window)