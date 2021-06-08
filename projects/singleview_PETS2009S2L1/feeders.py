import os
import sys
import numpy as np
import imageio
import uuid

import mot3d
from mot3d.utils import utils

class FeederPETS2009S2L1(object):
    
    def __init__(self, view='View_001', start=0, end=794, object_class='pedestrian', th_score=0.1, 
                 histograms={'enable':True, 'bins':255, 'channels':(0,1,2)},
                 bboxes={'enable':True}):
        self.view = view
        self.start = start
        self.end = end
        self.object_class = object_class
        self.th_score = th_score
        self.config_histograms = histograms
        self.config_bboxes = bboxes
        
        self.__Detection__ = mot3d.Detection2D
        
        self.classes_labels = {"pedestrian":1}
        self.folder = "/cvlabsrc1/cvlab/datasets_people_tracking/Crowd_PETS09/S2/L1/Time_12-34/{}".format(view)
        self.filenames = utils.find_files(self.folder, "frame*.jpg")[start:end]
        self.names = [os.path.basename(f) for f in self.filenames]
        
        base_dets = "./detections"
        self.detections = utils.json_read(os.path.join(base_dets,"fasterrcnn_resnet50_fpn_{}.json".format(view)))
        
        self.idx = -1
        
        self.image_size = self.get_image(0).shape[:3]        
        
    def get_image(self, idx):
        return imageio.imread(self.filenames[idx])
    
    def get_filename_image(self, idx):
        return self.filenames[idx]
    
    def get_bboxes(self, idx):
        name = self.names[idx]
        if name in self.detections['filenames']:
            i = self.detections['filenames'].index(name)
            bboxes = [[max(int(x),0) for x in box] for box in self.detections['bboxes'][i]]
            scores = self.detections['scores'][i]
            labels = self.detections['labels'][i]
        else:
            bboxes, scores, labels = [],[],[]
            
        return bboxes, scores, labels
    
    def __len__(self):
        return len(self.filenames)
        
    def __iter__(self):
        return self

    def __next__(self):
        
        self.idx += 1
        
        if self.idx==len(self):
            raise StopIteration
        
        name = self.names[self.idx]
        img = self.get_image(self.idx)
        
        detections = []
        indexes = []
        
        bboxes, scores, labels = self.get_bboxes(self.idx)
        
        for box,score,label in zip(bboxes, scores, labels):
            if label==self.classes_labels[self.object_class]:
                if score>self.th_score:
                    if len(box):

                        xmin, ymin, xmax, ymax = box                    
                        position = (xmax+xmin)/2, (ymax+ymin)/2

                        if self.config_histograms['enable']:
                            H = mot3d.color_histogram(img[ymin:ymax,xmin:xmax], 
                                                      channels=self.config_histograms['channels'], 
                                                      N=self.config_histograms['bins'])
                        else:
                            H = None

                        if self.config_bboxes['enable']:
                            bbox = (xmin, ymin, xmax, ymax)
                        else:
                            bbox = None

                        d = self.__Detection__(index=self.idx, 
                                               position=position, 
                                               confidence=score, 
                                               color_histogram=H, 
                                               bbox=bbox,
                                               id=uuid.uuid1(),
                                               view=self.view) 
                
                        detections.append(d)
                        indexes.append(self.idx)
                    
        return self.idx, detections

import sys, inspect
Feeder = {}
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj):
        Feeder[name] = obj