import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

import mot3d
from mot3d import utils

matplotlib.use('Agg')

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

    detections = []
    for idx_frame in range(start, stop):

        dets = [mot3d.Detection(d, 0.5, d) for d in detections_3d[idx_frame]]
        detections.append(dets)

    print("Building the graph..")
    
    def compute_weight(detection1, detection2, jump, dist):
        return [-1/jump * np.exp(-dist**2/20**2)]

    g = mot3d.build_graph(detections, p_source_sink_weights=0.01,
                          max_dist=300, max_jump=5, verbose=True,
                          compute_weight=compute_weight)
            
    print("-"*30)
    print("Solving the graph with muSSP..")
    print("-"*30)
    tracks, indexes, tracks_nodes = mot3d.solve_graph(g, verbose=True, method='muSSP')

    plt.figure()
    mot3d.plot_trajectories(tracks)
    plt.title("Output (muSSP)")
    plt.savefig("output_mussp.jpg")
    
    print("-"*30)
    print("Solving the graph with ILP..")
    print("-"*30)
    tracks, indexes, tracks_nodes = mot3d.solve_graph(g, verbose=True, method='ILP')

    plt.figure()
    mot3d.plot_trajectories(tracks)
    plt.title("Output (ILP)")
    plt.savefig("output_ilp.jpg")    