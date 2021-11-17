#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# ---------------------------------------------------------------------------
import sys
import os
import numpy as np
import itertools
from itertools import repeat
import multiprocessing
import threading
import time
from datetime import datetime
import uuid
import json
import imageio
import yaml
import argparse

import mot3d
from mot3d.utils import utils
from mot3d.utils import trajectory as utils_traj
from mot3d.weight_functions import (weight_distance_detections_2d, weight_confidence_detections_2d,\
                                    weight_distance_tracklets_2d, weight_confidence_tracklets_2d)   

from tracking_routines import *
import feeders
from scene import Scene

class Region(object):

    def __init__(self, filename):
        self.valid_region = imageio.imread(filename)
        if np.ndim(self.valid_region)>2:
            self.valid_region = self.valid_region[:,:,0]

    def is_inside(self, point):
        _point = np.reshape(point, (2,)).astype(np.int32)
        return self.valid_region[_point[1], _point[0]]!=0

class Tracker2D(object):
    
    def __init__(self, scene, valid_region, config, image_size):
        
        self.scene = scene
        self.valid_region = valid_region
        self.__c__ = config
        self.batch_size = self.__c__['batch_size']
        self.image_size = image_size

        __cfgd__ = self.__c__['detections']
        self.__Detection__ = mot3d.Detection2D
        
        weight_distance_d = lambda d1, d2 : weight_distance_detections_2d(d1, d2, **__cfgd__['weight_distance'])
        weight_confidence_d = lambda d : weight_confidence_detections_2d(d, **__cfgd__['weight_confidence'])        
        self.__compute_trajs__ = lambda detections: compute_trajectories(detections, 
                                                                         weight_source_sink=__cfgd__['weight_source_sink'], 
                                                                         weight_distance=weight_distance_d, 
                                                                         weight_confidence=weight_confidence_d,
                                                                         max_jump=__cfgd__['max_jump'],
                                                                         verbose=__cfgd__['verbose'])

        __cfgt__ = self.__c__['tracklets']
        self.__DetectionTracklet__ = lambda id,track,**kwargs: mot3d.DetectionTracklet2D(track, id=id, **kwargs, 
                                                                                         window=__cfgt__['length_endpoints'])

        weight_distance_t = lambda t1,t2 : weight_distance_tracklets_2d(t1, t2, **__cfgt__['weight_distance'])
        weight_confidence_t = lambda t : weight_confidence_tracklets_2d(t, **__cfgt__['weight_confidence'])         
        self.__cat_tracklets__ = lambda tracklets: concatenate_tracklets(tracklets,
                                                                         weight_source_sink=__cfgt__['weight_source_sink'], 
                                                                         weight_distance_tracklets=weight_distance_t,
                                                                         weight_confidence_tracklets=weight_confidence_t,
                                                                         max_jump=__cfgt__['max_jump'],
                                                                         verbose=__cfgt__['verbose'])   
        
    def update(self, time_index, detections):
        
        if self.valid_region is not None:
            # we filter the detections that are outside the valid region
            detections = [list(filter(lambda d: self.valid_region.is_inside(d.position), det_frame)) 
                          for det_frame in detections]
        
        # --------------------------------
        # compute tracklets for this batch
        # --------------------------------  
        
        print("[{}] Compute tracklets..".format(type(self).__name__))         
        tracklets = self.__compute_trajs__(detections)
        
        tracklet_splits = []
        for tracklet in tracklets:
            tracklet_splits += utils_traj.split_trajectory_modulo(tracklet, length=10)
            
        tracklet_splits = utils_traj.remove_short_trajectories(tracklet_splits, th_length=2)
        
        # --------------------------------------------------------------
        # concatenate the tracklets to the currently active trajectories
        # --------------------------------------------------------------
        
        print("[{}] Concatenate tracklets..".format(type(self).__name__))

        detections_tracklets =  [self.__DetectionTracklet__(id, track) for id, track in self.scene.active.items()]
        n_skipped = 0
        for tracklet in tracklet_splits:
            if self.valid_region is not None:
                res1 = self.valid_region.is_inside(tracklet[0].position) # tail
                res2 = self.valid_region.is_inside(tracklet[-1].position) # head
            else:
                res1, res2 = True, True
            
            if res1 and res2:
                det_tracklet = self.__DetectionTracklet__(None, tracklet,
                                                          tail_access_point=not res1, 
                                                          head_access_point=not res2)
                detections_tracklets.append(det_tracklet)
                self.scene.store_tracklet(tracklet)                
            else:
                # skip tracklets whose endpoints are boths in the access region
                n_skipped += 1 
                pass
                
        print("{} out of {} tracklets have been discarded.".format(n_skipped, len(tracklet_splits)))

        det_trajectories = self.__cat_tracklets__(detections_tracklets)

        for det_trajectory in det_trajectories:
            # if the id is not None it means that we have added new chunk(s) to one of the existing trajectories
            id = det_trajectory[0].id
            trajectory = utils_traj.concat_tracklets([dt.tracklet for dt in det_trajectory])
            trajectory = utils_traj.interpolate_trajectory(trajectory, attr_names=['position', 'bbox'])
            
            self.scene.update(id, trajectory)  
            
        self.scene.update_completed(time_index)

def main(config_file="main.config"):
    
    # ---------------------------------
    # Initialization
    # ---------------------------------
    __c__ = utils.yaml_read(config_file)
    
    output_path = __c__['output_path']
    utils.mkdir(output_path)
    
    feeder_name = __c__['feeder']['name']
    object_class = __c__['feeder'][feeder_name]['object_class']
    feeder = feeders.Feeder[feeder_name](**__c__['feeder'][feeder_name])
    iterator = iter(feeder)
    
    scene = Scene(__c__['tracking']['completed_after'])

    if __c__['tracking']['valid_region'] is not None and len(__c__['tracking']['valid_region']):
        valid_region = Region(__c__['tracking']['valid_region'])  
    else:
        valid_region = None

    tracker = Tracker2D(scene, valid_region, __c__['tracking'], image_size=feeder.image_size)

    times = {'overall':None, 'data collection':[], 'tracking':[]}
    start_time_all = time.time()
    
    # ---------------------------------
    # Looping over the data
    # ---------------------------------
    done = False
    while not done:
        
        start_time = time.time()
        
        # ---------------------------------
        # Acquire batch
        # ---------------------------------        
        indexes = []
        detections = []
        for _ in range(tracker.batch_size):
            try:
                _idx,_detections = next(iterator)
                indexes.append(_idx)
                detections.append(_detections)
            except StopIteration:
                done = True   
                break
        elapsed_time = time.time()-start_time
        times['data collection'].append(elapsed_time)
        print("Data collection time: {:0.3f}s".format(elapsed_time))
            
        # ---------------------------------
        # Compute trackle and connect to the previous ones
        # ---------------------------------             
        if len(detections):
            start_time = time.time()
            tracker.update(iterator.idx, detections) 
            elapsed_time = time.time()-start_time
            times['tracking'].append(elapsed_time)
            
            print("---Summary---")
            print("batch size: {}".format(len(detections)))
            print("indexes: {} -> {}".format(indexes[0], indexes[-1]))
            print("n_active:{} n_completed:{}".format(len(scene.active),
                                                        len(scene.completed)))
            print("Tracking time: {:0.3f}s".format(elapsed_time))
            print("-------------")

    elapsed_time_all = time.time()-start_time_all        
    times['overall'] = elapsed_time_all
    print("Overall time: {:0.2f}s".format(elapsed_time_all))   
    
    output_file = os.path.join(output_path, "times_{}.pickle".format(os.path.splitext(os.path.basename(config_file))[0]))
    utils.pickle_write(output_file, times)

    output_file = os.path.join(output_path, "results_{}.pickle".format(os.path.splitext(os.path.basename(config_file))[0]))
    scene.save(output_file)
    
    scene.active = {k:utils_traj.smooth_trajectory(x, s=2000, k=2, attr_names=['position','bbox']) 
                    for k,x in scene.active.items()}
    scene.completed = {k:utils_traj.smooth_trajectory(x, s=2000, k=2, attr_names=['position','bbox']) 
                       for k,x in scene.completed.items()}
    
    output_file = os.path.join(output_path, "results_smooth_{}.pickle".format(os.path.splitext(os.path.basename(config_file))[0]))
    scene.save(output_file)    

if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()   
    parser.add_argument("--config_file", "-c", type=str, default="config.config")

    args = parser.parse_args()

    main(**vars(args))

# python singleview_tracking.py -c config/config_singleview_PETS2009S2L1_View_001.yaml 