import numpy as np
import networkx as nx
import time
import os
import ast
import subprocess
import itertools
import imageio
from scipy import interpolate

from . import utils

__all__ = ["euclidean", "log_p", "sigmoid", "Detection", "merge_close_detections", 
           "build_graph", "solve_graph", "plot_trajectories", "visualisation", "smooth_trajectory",
           "interpolate_trajectory"]

def euclidean(p1, p2):
    return np.sqrt(sum([(i-j)**2 for i,j in zip(p1,p2)]))

def log_p(x):
    return -np.log(x/(1-x))

def sigmoid(x, a=5, b=0):
    return 1/(1+np.exp(-a*(x-b)))

class Detection(object):
    """
    Parameters
    ----------
    pos : list
        2d or 3d position
    confidence : float [0..1]
        detection confidence probability
    data : anything (optional)
        any other data you want to store
    w_entry_point : float (optional)
        weight of the edge connecting this node/detection to the source node
    w_exit_point : float (optional)
        weight of the edge connecting this node/detection to the sink node     
    """
    
    def __init__(self, pos, confidence=0.5, data=None, w_entry_point=None, w_exit_point=None):
        self.pos = pos
        self.confidence = confidence
        if not isinstance(data, list) and not None:
            self.data = [data]
        else:
            self.data = data
        self.w_entry_point=w_entry_point
        self.w_exit_point=w_exit_point
    
    def __str__(self):
        return """{self.__class__.__name__}(pos={self.pos}, confidence={self.confidence}, data={self.data}, w_entry_point={self.w_entry_point}, w_exit_point={self.w_exit_point})""".format(self=self)
    
def mean(detections):
    
    confidences = [ d.confidence for d in detections]
    positions = [ d.pos for d in detections]
    datas = [ d.data for d in detections]
    w_entry_points = [d.w_entry_point for d in detections if d.w_entry_point is not None]
    w_exit_points = [d.w_exit_point for d in detections if d.w_exit_point is not None]
    
    pos = np.average(positions, axis=0, weights=confidences)
    confidence = max(confidences)
    data = list(itertools.chain(*datas))
    w_entry_point = max(w_entry_points) if len(w_entry_points)>0 else None
    w_exit_point = max(w_exit_points) if len(w_exit_points)>0 else None
    
    return Detection(pos, confidence, data, w_entry_point, w_exit_point)

def merge_close_detections(detections, eps=0.3):
    from sklearn.cluster import DBSCAN
    
    if len(detections)==0:
        return detections
    
    points = np.array([ d.pos for d in detections])
    clu = DBSCAN(eps=eps, min_samples=1, metric='euclidean', algorithm='brute').fit(points)

    new_detections = []
    for i in set(clu.labels_):
        new_detections.append(mean([detections[j] for j in np.where(clu.labels_==i)[0]]))

    return new_detections
    
def build_graph(detections, p_source_sink_weights=0.1,
                max_dist=0.07, max_jump=12, verbose=True,
                compute_weight=None):
    """
    Build the MOT graph. The edge weights are function of 
    the distance between detections.
    
    Parameters
    ----------
    detections : list of lists of objects of type Detection
        in order of appearance from frame 0 to N.
    p_source_sink_weights : float, range [0..1]
        the probability associated to edges that connect to the source and sink nodes. 
        The edge weight is set to p_source_sink_weights*total_number_of_detections unless
        the detection has 'p_entry_point' (or 'p_exit_point' depending if it goes to source or sink).
        
        The algorithm produces more trajectories if p_source_sink_weights is small, fewer otherwise.
    max_dist : float
        an edge between two detections is created if the distance separating them is less 
        than jump*max_dist where jump is the difference in time.
    max_jump : int
        maximum difference in time between two detections
    """
    
    def p2w_entry_exit(x):
        return x#-log_p(sigmoid(x, 1125, 0))
    
    def p2w_pre_post(x):
        return log_p(sigmoid(x, 1, 0.5))
    
    def _compute_weight(detection1, detection2, jump, dist):
        return [-1/jump * np.exp(-dist**2/10**2)]
    
    if compute_weight is None:
        compute_weight = _compute_weight
    
    if verbose:
        from tqdm import tqdm
        _tqdm = tqdm
    else:
        _tqdm = lambda x: x

    # S->pre_i: represent how likely the detection i is the initial point 
    # pre_i->post_i: cost that reflects the reward of including the detection i
    # post_i->pre_j: encodes the similarity between detection i and j
    # post_i->T: represent how likely the detection i is the terminate point

    n_nodes = sum([len(x) for x in detections])
    if verbose: 
        print("Number of frames: {}".format(len(detections)))
        print("Number of detections: {}".format(n_nodes))

    SOURCE = 1
    SINK = int(n_nodes*2+2)
    source_sink_weight = n_nodes*p_source_sink_weights

    g = nx.DiGraph()
    g.add_node(SOURCE, label='source')
    g.add_node(SINK, label='sink')
    n = SOURCE+1

    # creates nodes for the detections, the edge going to source and sink and the pre->post edges
    m = 0
    for t, dets in enumerate(detections):
        if len(dets)>0:
            for i,d in enumerate(dets):
                
                # S->pre-nodes
                g.add_node(n, detection=d, time=t, label='pre-node', idet=m)
                if d.w_entry_point is not None:
                    g.add_edge(SOURCE, n, weight=d.w_entry_point)
                else:
                    g.add_edge(SOURCE, n, weight=source_sink_weight)
                    
                # post-nodes->T
                n += 1
                g.add_node(n, detection=d, time=t, label='post-node', idet=m)
                if d.w_exit_point is not None:
                    g.add_edge(n, SINK, weight=d.w_exit_point)   
                else:
                    g.add_edge(n, SINK, weight=source_sink_weight)
                    
                # pre-nodes->post-nodes
                g.add_edge(n-1, n, weight=p2w_pre_post(d.confidence))
                n += 1
                m += 1

    # creates the post-pre edges 
    i_resume = 0
    ndata = list(enumerate(g.nodes(data=True)))

    n_post_pre = 0
    for _,(n1,data1) in _tqdm(ndata):
        if data1['label']=='post-node':

            for i,(n2,data2) in ndata[i_resume:]:
                if data2['label']=='pre-node':

                    if data1['idet']!=data2['idet']:

                        jump = data2['time']-data1['time']
                        
                        # create edges that go forward in time only
                        if jump>0 and jump<=max_jump:

                            dist = euclidean(data1['detection'].pos, data2['detection'].pos)
                            '''
                            if dist<(max_dist*jump):
                                n_post_pre += 1
                                w = compute_weight(data1['detection'], data2['detection'], jump, dist)
                                g.add_edge(n1, n2, weight=w)   # post-nodes->pre-nodes
                            '''
                            ws = compute_weight(data1['detection'], data2['detection'], jump, dist)
                            if ws[0]<-0.1:
                                n_post_pre += 1
                                g.add_edge(n1, n2, weight=np.mean(ws))   # post-nodes->pre-nodes
                            

                        elif jump<=0:
                            i_resume = i
                        else:
                            # the nodes are ordered w.r.t the time so when we reach
                            # this point all the nodes that follow can be discarded as well.
                            # For this reason, we can break hear and continue. 
                            break

    if verbose:
        print("Number of post-pre nodes edges created: {}".format(n_post_pre))
        
    return g

def save_graph(g, filename="/tmp/graph.txt"):
    
    file = open(filename, "w") 
    file.write("p min {} {}\n".format(len(g), len(g.edges())))

    file.write("c ------ source->pre-nodes ------\n")
    for s,t,data in g.edges(data=True):
        if g.nodes[s]['label']=='source' and g.nodes[t]['label']=='pre-node':
            file.write("a {} {} {:0.7f}\n".format(s, t, data['weight']))

    file.write("c ------ post-nodes->sink ------\n")    
    for s,t,data in g.edges(data=True):
        if g.nodes[s]['label']=='post-node' and g.nodes[t]['label']=='sink':
            file.write("a {} {} {:0.7f}\n".format(s, t, data['weight']))        

    file.write("c ------ pre-node->post-nodes ------\n")    
    for s,t,data in g.edges(data=True):
        if g.nodes[s]['label']=='pre-node' and g.nodes[t]['label']=='post-node':
            file.write("a {} {} {:0.7f}\n".format(s, t, data['weight']))

    file.write("c ------ post-node->pre-nodes ------\n")    
    for s,t,data in g.edges(data=True):
        if g.nodes[s]['label']=='post-node' and g.nodes[t]['label']=='pre-node':
            file.write("a {} {} {:0.7f}\n".format(s, t, data['weight']))       

    file.close()

def interpolate_trajectory(track, time):
    """
    Fills the missing values in the trajectory using linear interpolation
    """

    track_inter = [track[0]]
    time_inter = [time[0]]
    for p,t in zip(track[1:], time[1:]):

        if (t-1)==time_inter[-1]:
            track_inter.append(p)
            time_inter.append(t)
        else:
            p1,p2 = track_inter[-1], p
            t1,t2 = time_inter[-1], t

            for s in range(1, t2-t1):
                dt = s/(t2-t1)
                if len(p1)==3:
                    p3 = (p1[0] + (p2[0]-p1[0])*dt, 
                            p1[1] + (p2[1]-p1[1])*dt,
                              p1[2] + (p2[2]-p1[2])*dt)
                elif len(p1)==2:
                    p3 = (p1[0] + (p2[0]-p1[0])*dt, 
                            p1[1] + (p2[1]-p1[1])*dt)                    
                t3 = t1+s

                track_inter.append(p3)
                time_inter.append(t3)

            track_inter.append(p)
            time_inter.append(t) 
            
    return np.array(track_inter), np.array(time_inter)

def smooth_trajectory(positions, n=1):
    """
    Spline smoothing
    """
    _positions = np.float32(positions)
    new_positions = [_positions[:n]]
    
    for i in range(n, len(positions)-n-1):
        
        window = _positions[i-n:i+n+1]
        window += np.random.rand(*window.shape)/1e6
        
        tck, u = interpolate.splprep(window.T, s=100, k=1)

        new_point = np.column_stack(interpolate.splev(0.5, tck))[0]
        
        new_positions.append(new_point)
        
    new_positions.append(_positions[-n-1:])
    
    return np.vstack(new_positions)

def _run_ssp(g, verbose=True, method='muSSP'):
    """
    Solve flow graph using Successive Shorthest Path (SSP) method.
    Very fast
    """
    
    input_filename = '/tmp/graph.txt'
    output_filename = '/tmp/output.txt'
    
    save_graph(g, input_filename)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))
    
    # solve graph
    cmd = os.path.join(curr_path, "{}/{} -i {} {}".format(method, method, input_filename, output_filename))
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                           universal_newlines=True, shell=True)
    
    if verbose:
        print(cmd)
        for line in out.stdout:
            print(line.strip())

    # this recovers the trajectories from the output file. (can be optimized)
    with open(output_filename, "r") as f:
        res_edges = [ [ int(xx) for xx in x.strip().split(' ')[1:]] for x in list(f)]

    g_copy = g.copy()
    for s,t,r in res_edges:
        if r==0:
            g_copy.remove_edge(s,t)
    SOURCE = list(g.nodes())[0]
    SINK = list(g.nodes())[-1]
    tracks_nodes = [ track[::2][1:] for track in list(nx.all_simple_paths(g_copy, SOURCE, SINK))] 
    
    return tracks_nodes

def _run_ilp(g, verbose=True):
    """
    Solve flow graph using Integer Lienar Programming method.
    Very slow
    """
    from .ilp import ilp
    
    tracks_nodes = ilp.solve_graph_ilp(g, verbose)
    
    return tracks_nodes

def solve_graph(g, verbose=True, method='muSSP'):
    
    if method in ['muSSP', 'FollowMe', 'SSP']:
        tracks_nodes = _run_ssp(g, verbose, method)
    elif method == 'ILP':
        tracks_nodes = _run_ilp(g, verbose)
    else:
        raise ValueError("Unrecognazed method '{}'. Choose 'muSSP' or 'ILP'")

    # transforms the trajectories from node IDs to positions + linear interpolatation
    tracks = []
    indexes = []
    n_discarded = 0
    for d in tracks_nodes:
        track = []
        index = []
        for n in d:
            node = g.nodes[n]
            track.append(node['detection'].pos)
            index.append(node['time'])
        track = np.array(track)
        index = np.array(index)

        track_inter, index_inter = interpolate_trajectory(track, index)

        tracks.append(track_inter.tolist())
        indexes.append(index_inter.tolist())    
    
    return tracks, indexes, tracks_nodes
            
colors = [[255,0,0], [0,255,0], 
          [100,100,255], [255,255,0], 
          [0,255,255], [255,0,255],
          [225,225,225], [0,0,0],
          [128,128,128], [50,128,50]]+[np.random.randint(0,255,3).tolist() for _ in range(200)]    
    
def plot_trajectories(tracks):
    import matplotlib.pyplot as plt
    plt.figure()
    for track,c in zip(tracks, colors):
        plt.plot(np.array(track)[:,0], np.array(track)[:,1], '.-', color=(c[0]/255,c[1]/255, c[2]/255), linewidth=2)
    plt.grid()
    
        
def visualisation(filenames, tracks, indexes, calibration=None, bboxes=None, 
                  crop=(slice(None,None), slice(None,None)), trace_length=25, thickness=5, thickness_boxes=2,
                  output_path="./output/sequence1", output_video="sequence1.mp4", fps=25):
    
    """
    Save all frames of a view with overlayed trajectories.
    
    Parameters
    ----------
    filenames : list of filenames (N,)
        list of image filenames in chronological order
    tracks : list of lists of positions
        the trajectories may have different lengths but with max length N.
        [
          [[x11,y11,z11],[x12,y12,z12],[x13,y13,z13], ...], # trajectory 1
          [              [x22,y22,z22],[x23,y23,z23], ...], # trajectory 2
          ...
        ]
    indexes : list of lists
        the trajectories indexes may have different lengths but in the range [0,N].
        [
          [t11, t12, t13, ...], # time for trajectory 1
          [     t22, t23, ...], # time for trajectory 2
          ...
        ] 
    calibration : dict
        extrinsic and instrinsic parameters {'R':.., 't':.., 'K':.., 'dist':..}        
    crop : list of slices
        to crop the image befor saving it
    trace_length : int
        the length of the overlayed trajectory in number of frames
    path_visuals : str
        output path
    """
    
    import cv2
    from tqdm import tqdm
    
    if crop is None:
        crop=(slice(None,None), slice(None,None))

    utils.mkdir(output_path) 

    for i,filename in tqdm(enumerate(filenames)):

        basename = os.path.basename(filename)
        img = imageio.imread(filename)

        for track,index,color in zip(tracks, indexes, colors):
            
            if i in index:

                track = np.float32(track)
                
                if calibration is not None:
                    K = np.array(calibration['K'])
                    dist = np.array(calibration['dist'])
                    R = np.array(calibration['R'])
                    rvec = cv2.Rodrigues(R)[0]
                    t = np.array(calibration['t'])

                    proj = cv2.projectPoints(track, rvec, t, K, dist)[0].reshape(-1,2)
                
                    track = np.int32(proj)
                ii = index.index(i)                
                

                t_init = np.maximum(ii-trace_length+1,0)
                for t0,t1 in zip(track[t_init:ii], track[t_init+1:ii+1]):
                    img = cv2.line(img, tuple(t0), tuple(t1), color=color, thickness=max(1, int(thickness*0.5)))

                img = cv2.circle(img, tuple(track[ii]), radius=thickness, color=color, thickness=-1)
                
        if bboxes is not None:
            for xmin, ymin, xmax, ymax in bboxes[i]:
                img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=[255,0,0], thickness=thickness_boxes)

        imageio.imwrite(os.path.join(output_path, basename), img[crop]) 
        
    if isinstance(output_video, str):
        ext = basename.split('.')[-1]
        cmd="ffmpeg -framerate {} -pattern_type glob -i '{}/*.{}' -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {} -y".format(fps, output_path, ext, output_video)
        print(cmd)
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
        for line in out.stdout:
            print(line.strip())   
    