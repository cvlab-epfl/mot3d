#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

import mot3d
from mot3d.utils import utils
import mot3d.utils.filt
from mot3d.utils import trajectory as utils_traj
import mot3d.weight_functions as wf

if __name__=="__main__":

    print("Loading detections..")
    detections_3d = utils.json_read("detections_3d_head.json")
    
    start = 49650
    stop = 50000      

    detections = []
    for index, index_frame in enumerate(range(start, stop)):
        
        # filtering out possible double detections.
        # This is not necessary if the double detections are sporadic.
        positions = detections_3d[str(index_frame)]
        positions_filtered = mot3d.utils.filt.merge_close_points(positions, eps=0.25)
        print("index:{} num positions:{} num positions after filtration:{}".format(index, len(positions), len(positions_filtered)))
        
        for position in positions_filtered:
            detections.append(mot3d.Detection3D(index, position))
        
    # The library supports manual setting of the entry and exit points. This allows to 
    # constrain where a trajectory can start and where it can end. By default all the detection
    # are entry and exit points.
    for d in detections:
        if d.index>(start+5) and d.index>(stop-5):
            d.entry_point = False
            d.exit_point = False
    
    print("Building graph on detections..")
    print("Appearance model disabled. We only use the positions..")
    
    weight_distance = lambda d1, d2: wf.weight_distance_detections_3d(d1, d2,
                                                                      sigma_jump=2, sigma_distance=0.15,
                                                                      sigma_color_histogram=0.3, sigma_box_size=0.3,
                                                                      max_distance=0.4,
                                                                      use_color_histogram=False, use_bbox=False)
    
    weight_confidence = lambda d: wf.weight_confidence_detections_3d(d, mul=1, bias=0)

    g = mot3d.build_graph(detections, weight_source_sink=50,
                          max_jump=8, verbose=True,
                          weight_confidence=weight_confidence,
                          weight_distance=weight_distance)
    if g is None:
        raise RuntimeError("There is not a single path between sink and source nodes!")
            
    print("-"*30)
    print("Solver first pass on detections..")
    print("-"*30)
    trajectories = mot3d.solve_graph(g, verbose=True, method='muSSP')
    
    tracklets = []
    for traj in trajectories:
        tracklets += mot3d.split_trajectory_modulo(traj, length=10)

    tracklets = mot3d.remove_short_trajectories(tracklets, th_length=2)
    
    plt.figure()
    mot3d.plot_trajectories(tracklets)
    plt.title("Extracted tracklets")
    plt.savefig("output_tracklets.jpg")    
        
    print("Building graph on tracklets..")
        
    detections_tracklets = [mot3d.DetectionTracklet3D(tracklet) for tracklet in tracklets]
    
    weight_distance_t = lambda t1, t2: wf.weight_distance_tracklets_3d(t1, t2, 
                                                                       sigma_color_histogram=0.3, sigma_motion=3, alpha=0.7,
                                                                       cutoff_motion=0.1, cutoff_appearance=0.1,
                                                                       max_distance=None,
                                                                       use_color_histogram=False)
    
    weight_confidence_t = lambda t: wf.weight_confidence_tracklets_3d(t, mul=1, bias=0)

    g = mot3d.build_graph(detections_tracklets, weight_source_sink=0.1,
                          max_jump=30, verbose=True,
                          weight_confidence=weight_confidence_t,
                          weight_distance=weight_distance_t)    
    if g is None:
        raise RuntimeError("There is not a single path between sink and source nodes!")       
    
    print("-"*30)
    print("Solver second pass on tracklets..")
    print("-"*30)
    trajectories = mot3d.solve_graph(g, verbose=True, method='muSSP')   
    
    # we have to concatenate the detections in each tracklet before interpolate!
    trajectories = [utils_traj.concat_tracklets([dt.tracklet for dt in traj]) for traj in trajectories]
    trajectories = list(map(lambda x: mot3d.interpolate_trajectory(x, attr_names=['position']), trajectories))    

    plt.figure()
    mot3d.plot_trajectories(trajectories)
    plt.title("Final result")
    plt.savefig("output.jpg")

    trajectories_smooth = list(map(lambda x: mot3d.smooth_trajectory(x, s=1, attr_names=['position']), trajectories))
    
    plt.figure()
    mot3d.plot_trajectories(trajectories_smooth)
    plt.title("Final result smooth")
    plt.savefig("output_smooth.jpg")

    print("Save images with overlayed trajectories and create a video..")
    print("This requires FFmpeg!")
    calibration = utils.json_read("camera_pose.json")
    filenames = ["frames/frame_{:05d}.jpg".format(idx) for idx in range(start, start+50)]
    
    positions, indexes = [],[]
    for trajectory in trajectories_smooth:
        _positions, _indexes = zip(*[(d.position, d.index) for d in trajectory])
        positions.append(_positions)
        indexes.append(_indexes)

    mot3d.visualisation(filenames, positions, indexes, calibration, 
                        crop=(slice(250,750), slice(800,1200)), trace_length=15, 
                        output_path="./output/sequence1", output_video="./output/sequence1.mp4")  