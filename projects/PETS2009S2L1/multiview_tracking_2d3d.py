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
import copy
import yaml
import argparse

import mot3d
from mot3d.utils import utils
from mot3d.utils import trajectory as utils_traj
from mot3d.types import DetectionTracklet2D, DetectionTracklet3D, TrajectoryView
from tracking_routines import *
import feeders
from scene import Scene, Scene2d3d

from multiview_tracking import MultiviewMatcher, Tracker3D
from singleview_tracking import Tracker2D
from multiview_tracking import Region as RegionPolygon
from singleview_tracking import Region as RegionImage 
        
import cv2
def project_points(points, calib):
    R = np.float32(calib['R'])
    rvec = cv2.Rodrigues(R)[0]
    tvec = np.float32(calib['t'])
    K = np.float32(calib['K'])
    dist = np.float32(calib['dist'])
    
    points = np.reshape(points, (-1,3), np.float32)
    return cv2.projectPoints(points, rvec, tvec, K, dist)[0].reshape(-1,2)

def project_trajectory(trajectory, calib):
    indices, positions = list(zip(*[(d.index, d.position) for d in trajectory]))
    proj = project_points(positions, calib)
    return [mot3d.Detection2D(i, p) for i,p in zip(indices, proj)]    

def recover_trajectory_2d(trajectory, trajectories_3d, calib):
    chunks = []
    chunk = []
    id_views = []
    for i,d in enumerate(trajectory):
        if isinstance(d, TrajectoryView):
            chunks.append(chunk)
            chunk = []
            
            traj3d_proj = project_trajectory(trajectories_3d[d.id_view], calib)
            chunks.append(traj3d_proj)
            id_views.append(d.id_view)
        else:
            chunk.append(d)

    chunks.append(chunk)
            
    return list(itertools.chain(*chunks)), id_views    
        
class Tracker2D3D(Tracker2D):
    
    def __init__(self, scene, valid_region, config, image_size, calibration):
        super(Tracker2D3D, self).__init__(scene, valid_region, config, image_size)
        
        self.calibration = calibration
        
    def __DetectionTrackletPlus__(self, id, track):
        '''
        track may contain objects of type TrajectoryView. This object acts as pointer to trajectories stored
        somewhere else. When we create a new DetectionTracklet we have to reconstruct the 
        trajectory by projecting the 3d ones into this view.
        id_views is a list that contains all the ids of the 3D trajectories in this track.
        '''
        track_2d, id_views = recover_trajectory_2d(track, 
                                                   {**self.scene.scene3d.active, **self.scene.scene3d.completed}, 
                                                   self.calibration)

        d = self.__DetectionTracklet__(id, track_2d)
        d.tracklet_orig = track
        d.id_views = id_views  
        return d, id_views
        
    def update(self, time_index, detections_2d):
        
        # --------------------------------
        # compute tracklets for this batch
        # --------------------------------  
        
        print("[{}] Compute tracklets..".format(type(self).__name__))        
        tracklets = self.__compute_trajs__(detections_2d)
        
        tracklet_splits = []
        for tracklet in tracklets:
            tracklet_splits += utils_traj.split_trajectory_modulo(tracklet, length=10)
            
        tracklet_splits = utils_traj.remove_short_trajectories(tracklet_splits, th_length=2)
        
        # --------------------------------------------------------------
        # concatenate the tracklets to the currently active trajectories
        # --------------------------------------------------------------
        '''
        Here we concatenate 2D and 3D trajectories. The 3D trajectories are stored in scene3d and are computed by the Tracker3D.
        We assume that once a 3D trajectory is assigned an id it will never change.
        Since we have divided 2D and 3D detections before tracking, there is no overlap or duplicate trajectories between 
        the 3D scene and the 2D scene. As a result, we can simply concatenate these trajecotries over time.
        To create a mixed 2D 3D trajectoriy we use an object called TrajectoryView. It acts as pointer to a trajectory stored
        in the 3D scene. A 2D trajectory can for example be: [Detection2D, Detection2D, ..., TrajectoryView(id_view), ..., Detection2D]
        Every time we have to use this 2D trajectory we have/could recompose it by projecting the 3D trajectory pointed 
        by TrajectoryView in this view (or we could simply project the tail and head of the trajectory as only tese are 
        used during the matching process).
        '''
        
        print("[{}] Concatenate tracklets..".format(type(self).__name__))

        # picking the active trajectories in this view
        ids_3d_not_to_pick = []
        detections_tracklets = []
        for id, track in self.scene.active.items():

            d, id_views = self.__DetectionTrackletPlus__(id, track)
                
            detections_tracklets.append(d)
                
            # these are all the ids of 3D trajectories that are somwhere in the middle/part of these 2D track.
            # These 3D trajectories should not be added again in detections_tracklets.
            ids_3d_not_to_pick += id_views#[d.id_view for i,d in enumerate(track) if isinstance(d, TrajectoryView)]
                
        # picking active 3D trajectories by exlcuding the ones we already have seen
        for id_view, _ in self.scene.scene3d.active.items():           
            if id_view not in ids_3d_not_to_pick:
                
                d, _ = self.__DetectionTrackletPlus__(None, [TrajectoryView(id_view)])
                
                detections_tracklets.append(d)
        
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
                det_tracklet.tracklet_orig = tracklet
                detections_tracklets.append(det_tracklet)
                #self.scene.store_tracklet(tracklet)      
                
            else:
                # skip tracklets whose endpoints are boths in the access region
                n_skipped += 1 
                pass
                
        print("{} out of {} tracklets have been discarded.".format(n_skipped, len(tracklet_splits)))
        
        det_trajectories = self.__cat_tracklets__(detections_tracklets)
        
        for det_trajectory in det_trajectories:
            # det_trajectory contains a chain of mixed DetectionTracklet2D and/or TrajectoryView objects with or without id.
            # If the id of the first element is None, scene.update(.) will create a new trajectory. If id is not None
            # it means that scene has already seen that trajectoriy so it will simply update its state.

            id = det_trajectory[0].id
            
            trajectory = []
            for dt in det_trajectory:
                trajectory.append(dt.tracklet_orig)
                
            trajectory = utils_traj.concat_tracklets(trajectory)
            
            # cannot be done here as trajectory may contain TrajectoryView objects
            #trajectory = utils_traj.interpolate_trajectory(trajectory, attr_names=['position', 'bbox'])
            
            self.scene.update(id, trajectory)
              
        self.scene.update_completed(time_index)        

def main(config_file="main.config"):
    
    # ---------------------------------
    # Initialization
    # ---------------------------------
    __c__ = utils.yaml_read(config_file)
    
    views_2d3d = __c__['views_2d3d']    
    
    output_path = __c__['output_path']
    utils.mkdir(output_path)    
    
    views = __c__['views']
    feeder_name = __c__['feeder']['name']
    object_class = __c__['feeder'][feeder_name]['object_class']
    feeder = {view:feeders.Feeder[feeder_name](view, **__c__['feeder'][feeder_name]) for view in views}
    iterator = {view:iter(feeder[view]) for view in views}
    
    __c3d__ = utils.yaml_read(__c__['configs']['3D'])
    
    calibration = utils.json_read(__c3d__['calibration'])
    
    # setting up the 3D region and the projections
    data = utils.json_read(__c3d__['valid_regions'])
    valid_regions_3d = {}
    valid_regions_3d['ground'] = RegionPolygon(data['ground'])
    for view in views: 
        valid_regions_3d[view] = RegionPolygon(data[view])
        
    # setting up the 2D regions
    valid_region_2d = {}
    for view in views_2d3d:
        config_view = utils.yaml_read(__c3d__['configs'][view])
        valid_region_2d[view] = RegionImage(config_view['tracking']['valid_region'])        
        
    # setting up the scenes, one for 3D and one for each view
    scenes = {}
    scenes['3D'] = Scene(__c3d__['tracking']['completed_after'])   
    for view in views_2d3d:
        config_view = utils.yaml_read(__c3d__['configs'][view])
        scenes[view] = Scene2d3d(scenes['3D'], config_view['tracking']['completed_after'])    

    matcher = MultiviewMatcher(views, calibration, __c3d__['matching'])
    
    # setting up the trackers
    trackers = {}
    trackers['3D'] = Tracker3D(scenes['3D'], valid_regions_3d, __c3d__['tracking'])
    for view in views_2d3d:
        config_view = utils.yaml_read(__c3d__['configs'][view])
        trackers[view] = Tracker2D3D(scenes[view], 
                                     valid_region_2d[view], 
                                     config_view['tracking'], 
                                     image_size=feeder[view].image_size,
                                     calibration=calibration[view])

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
        for _ in range(trackers['3D'].batch_size):
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
            detections_outside = [{view:[] for view in views} for _ in range(len(detections))]
            for i,_detections in enumerate(detections):
                for view,ds in _detections.items():
                    for d in ds:
                        if valid_regions_3d[view].is_inside(d.position):
                            detections_inside[i][view].append(d)
                        else:
                            detections_outside[i][view].append(d)
            elapsed_time = time.time()-start_time
            print("Inside/outside detection separation time: {:0.3f}s".format(elapsed_time))
            
            idx = iterator[views[0]].idx
            
            start_time = time.time()
            detections_3d, unused_detections_2d = matcher.compute(detections_inside) 
            elapsed_time = time.time()-start_time
            times['matching'].append(elapsed_time)
            print("Matching + triangulation time: {:0.3f}s".format(elapsed_time))
            
            # We exclude the 3D detections that appears to be outside the valid ground region.
            # we use them as 2D detections instead.
            detections_3d_ = []
            for i,ds in enumerate(detections_3d):
                ds_ = []
                for d in ds:
                    if valid_regions_3d['ground'].is_inside(d.position):
                        ds_.append(d)
                    else:
                        for view,d2d in d.detections_2d.items():
                            unused_detections_2d[i][view].append(d2d)
                detections_3d_.append(ds_)
            detections_3d = detections_3d_
            
            start_time = time.time()
            trackers['3D'].update(idx, detections_3d) 
            # this loop can be executed in parallel
            for view in views_2d3d:
                detections_2d = [do[view]+du[view] for do,du in zip(detections_outside,unused_detections_2d)]
                trackers[view].update(idx, detections_2d) 
            elapsed_time = time.time()-start_time
            times['tracking'].append(elapsed_time)
            print("Overall tracking time: {:0.3f}s".format(elapsed_time))            
            
            print("---Summary---")
            print("batch size: {}".format(len(detections)))
            print("indexes: {} -> {}".format(indexes[0], indexes[-1]))
            print("[3D] n_active:{} n_completed:{}".format(len(scenes['3D'].active), len(scenes['3D'].completed)))
            for view in views_2d3d:
                print("[{}] n_active:{} n_completed:{}".format(view, len(scenes[view].active), len(scenes[view].completed)))
            print("-------------")

    elapsed_time_all = time.time()-start_time_all        
    times['overall'] = elapsed_time_all
    print("Overall time: {:0.2f}s".format(elapsed_time_all)) 
    
    output_file = os.path.join(output_path, "times_{}.pickle".format(os.path.splitext(os.path.basename(config_file))[0]))
    utils.pickle_write(output_file, times)

    output_file = os.path.join(output_path, "results_2d3d_{}_{}.pickle".format('{}',os.path.splitext(os.path.basename(config_file))[0]))
    scenes['3D'].save(output_file.format('3D'))
    for view in views_2d3d:
        scenes[view].save(output_file.format(view))

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

# python multiview_tracking_2d3d.py -c config/config_2d3d_PETS2009S2L1.yaml