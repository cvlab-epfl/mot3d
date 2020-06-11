import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

import mot3d
from mot3d import utils

matplotlib.use('Agg')

if __name__=="__main__":

    print("Loading detections..")
    detections_3d = utils.json_read("detections_3d_head_test_25fps_0.05_49500_50000.json")

    print("Merging close detections..")
    start = 49510
    stop = 50000     
    detections = []
    for idx_frame in range(start, stop):

        dets = [mot3d.Detection(d['pos'], 0.5, d) for d in detections_3d[str(idx_frame)]]
        new_dets = mot3d.merge_close_detections(dets, eps=0.4)

        detections.append(new_dets)
        
    '''
    # The algorithm supports manual setting of the entry and exit weights for each detection.
    # If you know the number of objects in the scene you may decide to manually
    # annotate the first and last frame. Then set the weights of entry and exit edges to a low value 
    # and a high value everywhere else. As a result, the algorithm is encouraged to produce 
    # the wanted number of trajectories.
    
    for i,dets in enumerate(detections):
        for d in dets:
            d.w_entry_point = 999999.0
            d.w_exit_point = 999999.0
    for d in detections[0]:
        d.w_entry_point = 0.0
    for d in detections[-1]:
        d.w_exit_point = 0.0      
    '''
    
    print("Building the graph..")
    g = mot3d.build_graph(detections, p_source_sink_weights=0.1,
                          max_dist=0.07, max_jump=12, verbose=True,
                          compute_weight=None)   

    print("Solving the graph..")
    tracks, indexes = mot3d.solve_graph(g, verbose=True)

    print("Save trajectories plot..")
    mot3d.plot_trajectories(tracks)
    plt.savefig("trajectories.jpg")

    smooth_tracks = list(map(lambda x: mot3d.smooth_trajectory(x, n=10), tracks))
    mot3d.plot_trajectories(smooth_tracks)
    plt.savefig("trajectories_smooth.jpg")

    print("Save images with overlayed trajectories and create a video..")
    print("This requires FFmpeg!")
    calibration = utils.json_read("camera_pose.json")
    filenames = ["frames/frame_{:05d}.jpg".format(idx) for idx in range(start, start+50)]

    mot3d.visualisation(filenames, smooth_tracks, indexes, calibration, 
                        crop=(slice(250,750), slice(800,1200)), trace_length=25, 
                        output_path="./output/sequence1", output_video="./output/sequence1.mp4")