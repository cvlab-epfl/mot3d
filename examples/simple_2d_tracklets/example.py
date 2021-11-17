#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import mot3d
import mot3d.weight_functions as wf
from mot3d.utils import utils

if __name__=="__main__":

    print("Loading detections..")

    ps1 = np.array(utils.json_read("track_line1.json"))
    ps2 = np.array(utils.json_read("track_line2.json"))

    ps2 = ps2-np.array([[4,0]])

    ps1[4] = ps1[4]+np.random.randn(2)
    ps2[4] = ps2[4]+np.random.randn(2)

    plt.figure()
    plt.plot(ps1[:,0], ps1[:,1], 'r.')
    plt.plot(ps2[:,0], ps2[:,1], 'b.')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title("Input detections")
    plt.savefig("input.jpg")

    detections_3d = list(zip(ps1, ps2))

    start = 0
    stop = len(detections_3d)
    
    dummy_color_histogram = [np.ones((128,))]*3
    dummy_bbox = [0,0,1,1]

    detections = []
    for index in range(start, stop):
        for position in detections_3d[index]:
            detections.append(mot3d.Detection2D(index, position, 
                                                color_histogram=dummy_color_histogram,
                                                bbox=dummy_bbox))

    print("Building graph on detections..")
    
    weight_distance = lambda d1, d2: wf.weight_distance_detections_2d(d1, d2,
                                                                      sigma_jump=1, sigma_distance=10,
                                                                      sigma_color_histogram=0.3, sigma_box_size=0.3,
                                                                      max_distance=15,
                                                                      use_color_histogram=True, use_bbox=True)
    
    weight_confidence = lambda d: wf.weight_confidence_detections_2d(d, mul=1, bias=0)

    g = mot3d.build_graph(detections, weight_source_sink=0.1,
                          max_jump=4, verbose=True,
                          weight_confidence=weight_confidence,
                          weight_distance=weight_distance)
    if g is None:
        raise RuntimeError("There is not a single path between sink and source nodes!") 
        
    plt.figure()
    mot3d.plot_graph(g)
    plt.title("Min-cost max-flow graph on detections")
    plt.savefig("graph_detections.jpg")        
            
    print("-"*30)
    print("Solver first pass on detections..")
    print("-"*30)
    trajectories = mot3d.solve_graph(g, verbose=True, method='muSSP')  
    
    tracklets = []
    for traj in trajectories:
        tracklets += mot3d.split_trajectory_modulo(traj, length=5)

    tracklets = mot3d.remove_short_trajectories(tracklets, th_length=2)

    plt.figure()
    mot3d.plot_trajectories(tracklets)
    plt.title("Extracted tracklets")
    plt.savefig("output_tracklets.jpg") 
        
    print("Building graph on tracklets..")
        
    detections_tracklets = [mot3d.DetectionTracklet2D(tracklet) for tracklet in tracklets]
    
    weight_distance_t = lambda t1, t2: wf.weight_distance_tracklets_2d(t1, t2, max_distance=None,
                                                                       sigma_color_histogram=0.3, sigma_motion=50, alpha=0.7,
                                                                       cutoff_motion=0.1, cutoff_appearance=0.1,
                                                                       use_color_histogram=True)
    
    weight_confidence_t = lambda t: wf.weight_confidence_tracklets_2d(t, mul=1, bias=0)

    g = mot3d.build_graph(detections_tracklets, weight_source_sink=0.1,
                          max_jump=20, verbose=True,
                          weight_confidence=weight_confidence_t,
                          weight_distance=weight_distance_t)    
    if g is None:
        raise RuntimeError("There is not a single path between sink and source nodes!") 
        
    plt.figure()
    mot3d.plot_graph(g)
    plt.title("Min-cost max-flow graph on tracklets")
    plt.savefig("graph_tracklets.jpg")        
    
    print("-"*30)
    print("Solver second pass on tracklets..")
    print("-"*30)
    trajectories = mot3d.solve_graph(g, verbose=True, method='muSSP')    

    plt.figure()
    mot3d.plot_trajectories(trajectories)
    plt.title("Final result")
    plt.savefig("output.jpg")
      