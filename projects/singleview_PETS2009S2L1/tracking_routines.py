import sys
import os
import numpy as np
import copy
from collections import Iterable
import itertools

import multiview_matching as mm
import mot3d
from mot3d.utils import trajectory as utils_traj

def compute_trajectories(detections, weight_source_sink, 
                         weight_distance, weight_confidence,
                         max_jump=4, verbose=False):
    
    detections = list(itertools.chain(*detections))

    g = mot3d.build_graph(detections, 
                          weight_source_sink=weight_source_sink,
                          max_jump=max_jump, 
                          verbose=verbose, 
                          weight_distance=weight_distance,
                          weight_confidence=weight_confidence)   

    if g is None:
        trajectories = [] 
    else:
        trajectories = mot3d.solve_graph(g, verbose=verbose)
               
    return trajectories

def concatenate_tracklets(detections_tracklets, weight_source_sink, 
                          weight_confidence_tracklets, weight_distance_tracklets,
                          max_jump=100, verbose=False):
    
    g = mot3d.build_graph(detections_tracklets, 
                          weight_source_sink=weight_source_sink,
                          max_jump=max_jump, 
                          verbose=verbose,
                          weight_confidence=weight_confidence_tracklets,
                          weight_distance=weight_distance_tracklets)

    if g is None:
        trajectories_tracklets = []
    else:
        trajectories_tracklets = mot3d.solve_graph(g, verbose=verbose)      
        
    return trajectories_tracklets