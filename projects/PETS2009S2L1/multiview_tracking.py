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
from shapely.geometry import Polygon, Point

import mot3d
from mot3d.utils import utils
from mot3d.utils import trajectory as utils_traj
from mot3d.weight_functions import (weight_distance_detections_3d, weight_confidence_detections_3d,\
                                    weight_distance_tracklets_3d, weight_confidence_tracklets_3d) 

from tracking_routines import *
import feeders
from scene import Scene

class Region(object):

    def __init__(self, polygon):
        self.p = Polygon(polygon)

    def is_inside(self, point):
        return self.p.contains(Point(*point[:2]))

class MultiviewMatcher(object):
    
    def __init__(self, views, calibration, config):
        self.views = views
        self.calibration = calibration        
        self.__c__ = config
        
    def compute(self, detections_2d):
        
        pool = multiprocessing.Pool(self.__c__['threads'])
        res = pool.starmap(multiview_matching_parallel, zip(detections_2d, 
                                                           repeat(self.views),
                                                           repeat(self.calibration),
                                                           repeat(self.__c__['max_distance']),
                                                           repeat(self.__c__['distance_none']),
                                                           repeat(self.__c__['n_candidates']),
                                                           repeat(self.__c__['n_min_views_clique']),
                                                           repeat(self.__c__['verbose'])))
        detections_3d, remaining_detections_2d = list(zip(*res))
        
        pool.close()
        pool.join() 
        
        return detections_3d, remaining_detections_2d

class Tracker3D(object):
    
    def __init__(self, scene, valid_regions, config):
        
        self.scene = scene
        self.valid_regions = valid_regions
        self.__c__ = config
        self.batch_size = self.__c__['batch_size']

        __cfgd__ = self.__c__['detections']
        self.__Detection__ = mot3d.Detection3D
        
        weight_distance_d = lambda d1, d2 : weight_distance_detections_3d(d1, d2, **__cfgd__['weight_distance'])
        weight_confidence_d = lambda d : weight_confidence_detections_3d(d, **__cfgd__['weight_confidence'])        
        self.__compute_trajs__ = lambda detections: compute_trajectories(detections, 
                                                                         weight_source_sink=__cfgd__['weight_source_sink'], 
                                                                         weight_distance=weight_distance_d, 
                                                                         weight_confidence=weight_confidence_d,
                                                                         max_jump=__cfgd__['max_jump'],
                                                                         verbose=__cfgd__['verbose'])

        __cfgt__ = self.__c__['tracklets']
        self.__DetectionTracklet__ = lambda id,track,**kwargs: mot3d.DetectionTracklet3D(track, id=id, **kwargs, 
                                                                                         window=__cfgt__['length_endpoints'])

        weight_distance_t = lambda t1,t2 : weight_distance_tracklets_3d(t1, t2, **__cfgt__['weight_distance'])
        weight_confidence_t = lambda t : weight_confidence_tracklets_3d(t, **__cfgt__['weight_confidence'])         
        self.__cat_tracklets__ = lambda tracklets: concatenate_tracklets(tracklets,
                                                                         weight_source_sink=__cfgt__['weight_source_sink'], 
                                                                         weight_distance_tracklets=weight_distance_t,
                                                                         weight_confidence_tracklets=weight_confidence_t,
                                                                         max_jump=__cfgt__['max_jump'],
                                                                         verbose=__cfgt__['verbose']) 
        
    def update(self, time_index, detections_3d):       
        
        # --------------------------------
        # compute tracklets for this batch
        # --------------------------------
        
        print("[{}] Compute tracklets..".format(type(self).__name__))  
        start_time = time.time()        
        tracklets = self.__compute_trajs__(detections_3d)
        elapsed_time = time.time()-start_time
        print("Tracking detections time: {:0.3f}s".format(elapsed_time))        
        
        start_time = time.time()
        tracklet_splits = []
        for tracklet in tracklets:
            #tracklet = utils_traj.interpolate_trajectory(tracklet, attr_names=['position'])
            tracklet_splits += utils_traj.split_trajectory_modulo(tracklet, length=5)
            
        tracklet_splits = utils_traj.remove_short_trajectories(tracklet_splits, th_length=2)
        elapsed_time = time.time()-start_time
        print("Spliting time: {:0.3f}s".format(elapsed_time)) 

        # --------------------------------------------------------------
        # concatenate the tracklets to the currently active trajectories
        # --------------------------------------------------------------
        
        start_time = time.time()
        print("[{}] Concatenate tracklets..".format(type(self).__name__))
        detections_tracklets =  [self.__DetectionTracklet__(id, track) for id, track in self.scene.active.items()]
        n_skipped = 0
        for tracklet in tracklet_splits:
            if self.valid_regions['ground'] is not None:
                res1 = self.valid_regions['ground'].is_inside(tracklet[0].position) # tail
                res2 = self.valid_regions['ground'].is_inside(tracklet[-1].position) # head
            else:
                res1, res2 = True, True
            
            if res1 and res2:
                det_tracklet = self.__DetectionTracklet__(None, tracklet,
                                                          tail_access_point=not res1, 
                                                          head_access_point=not res2)
                detections_tracklets.append(det_tracklet)
                self.scene.store_tracklet(tracklet)
            else:            
                # skip tracklets whose endpoints are boths outside the valid region
                n_skipped += 1
                pass
                
        print("{} out of {} tracklets have been discarded.".format(n_skipped, len(tracklet_splits)))
        elapsed_time = time.time()-start_time
        print("Tracklet processing time: {:0.3f}s".format(elapsed_time))

        start_time = time.time()
        det_trajectories = self.__cat_tracklets__(detections_tracklets)
        elapsed_time = time.time()-start_time
        print("Concatenation time: {:0.3f}s".format(elapsed_time))         

        start_time = time.time()        
        for det_trajectory in det_trajectories:
            # if the id is not None it means that we have added new chunk(s) to one of the existing trajectories
            id = det_trajectory[0].id
            trajectory = utils_traj.concat_tracklets([dt.tracklet for dt in det_trajectory])
            trajectory = utils_traj.interpolate_trajectory(trajectory, attr_names=['position'])
            
            self.scene.update(id, trajectory) 
        elapsed_time = time.time()-start_time
        print("Scene update time: {:0.3f}s".format(elapsed_time))            
            
        self.scene.update_completed(time_index)    

def main(config_file="main.config"):
    
    # ---------------------------------
    # Initialization
    # ---------------------------------
    __c__ = utils.yaml_read(config_file)
    
    output_path = __c__['output_path']
    utils.mkdir(output_path)
    
    views = __c__['views']
    
    feeder_name = __c__['feeder']['name']
    object_class = __c__['feeder'][feeder_name]['object_class']
    feeder = {view:feeders.Feeder[feeder_name](view, **__c__['feeder'][feeder_name]) for view in views}
    iterator = {view:iter(feeder[view]) for view in views}
    
    calibration = utils.json_read(__c__['calibration'])
    
    scene = Scene(__c__['tracking']['completed_after'])
    
    data = utils.json_read(__c__['valid_regions'])
    valid_regions = {'ground':Region(data['ground'])}
    for view in views: 
        valid_regions[view] = Region(data[view])

    matcher = MultiviewMatcher(views, calibration, __c__['matching'])
    tracker = Tracker3D(scene, valid_regions, __c__['tracking'])

    times = {'overall':None, 'data collection':[], 'matching':[], 'tracking':[]}
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
                _idxs,_detections = list(zip(*[next(iterator[view]) for view in views]))
                indexes.append(_idxs[0])
                detections.append({view:d for view,d in zip(views,_detections)})
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
            
            # we divide the detections that should be used for 3D tracking from the rest.
            start_time = time.time()
            detections_inside = [{view:[] for view in views} for _ in range(len(detections))]
            for i,_detections in enumerate(detections):
                for view,ds in _detections.items():
                    for d in ds:
                        if valid_regions[view].is_inside(d.position):
                            detections_inside[i][view].append(d)
            elapsed_time = time.time()-start_time
            print("Inside/outside detection separation time: {:0.3f}s".format(elapsed_time))
            
            idx = iterator[views[0]].idx
            
            start_time = time.time()
            detections_3d, _ = matcher.compute(detections_inside) 
            elapsed_time = time.time()-start_time
            times['matching'].append(elapsed_time)
            print("Matching + triangulation time: {:0.3f}s".format(elapsed_time))
            
            # We exclude the 3D detections that appears to be outside the valid ground region.
            detections_3d_ = []
            for i,ds in enumerate(detections_3d):
                ds_ = [d for d in ds if valid_regions['ground'].is_inside(d.position)]
                detections_3d_.append(ds_)
            detections_3d = detections_3d_           
            
            start_time = time.time()
            tracker.update(idx, detections_3d) 
            elapsed_time = time.time()-start_time
            times['tracking'].append(elapsed_time)
            print("Overall tracking time: {:0.3f}s".format(elapsed_time))            
            
            print("---Summary---")
            print("batch size: {}".format(len(detections)))
            print("indexes: {} -> {}".format(indexes[0], indexes[-1]))
            print("n_active:{} n_completed:{}".format(len(scene.active), len(scene.completed)))
            print("-------------")

    elapsed_time_all = time.time()-start_time_all        
    times['overall'] = elapsed_time_all
    print("Overall time: {:0.2f}s".format(elapsed_time_all))  
    
    output_file = os.path.join(output_path, "times_{}.pickle".format(os.path.splitext(os.path.basename(config_file))[0]))
    utils.pickle_write(output_file, times)    

    output_file = os.path.join(output_path, "results_{}.pickle".format(os.path.splitext(os.path.basename(config_file))[0]))
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

# python multiview_tracking.py -c config/config_multiview_PETS2009S2L1.yaml 