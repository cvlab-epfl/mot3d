import numpy as np
import networkx as nx
import time
import os
import ast
import copy
import subprocess
import itertools
import imageio
import functools
from scipy import interpolate
import matplotlib.pyplot as plt
from collections import Iterable

from . import utils
from .features import *
from .types import *

__all__ = ["euclidean", "build_graph", "solve_graph", 
           "plot_trajectories", "visualisation", "smooth_trajectory",
           "interpolate_trajectory", "smooth_polynomial_2d", "plot_graph",
           "cost_trajectory", "merge_close_points", "one_to_one_matching",
           "concat_tracklets"]     
        
def euclidean(p1, p2):
    assert len(p1)==len(p2)
    return np.sqrt(sum([(i-j)**2 for i,j in zip(p1,p2)]))        
        
def weight_distance(jump, distance, detection1=None, detection2=None,
                    sigma_jump=3, sigma_distance=2):
    """
    Computes the weight of the edges linking two detections

    Parameters
    ----------
    jump : int
        delta time between the this and another detection
    distance : float
        spatial distance between this and another detection
    detection1, detection2 : Detection
        use these if you use a custom Detection object       
    """
    return -np.exp(-(jump-1)**2/sigma_jump**2) * np.exp(-distance**2/sigma_distance**2)

def weight_confidence(detection):
    """
    Computes the weight of the edges linking the pre-node and post-node
    """
    return -detection.confidence      

def weight_confidence_trackelts(tracklet):
    return -0.11

def merge_close_points(points, eps=0.3, return_indexes=False):
    from sklearn.cluster import DBSCAN
    
    if len(points)==0:
        return points
    
    points_ = np.array(points)
    
    clu = DBSCAN(eps=eps, min_samples=1, metric='euclidean', algorithm='brute').fit(points_)

    new_points = []
    for i in set(clu.labels_):
        if return_indexes:
            new_points.append(np.where(clu.labels_==i)[0].tolist())
        else:
            new_points.append(np.mean([points_[j] for j in np.where(clu.labels_==i)], axis=1))
    if not return_indexes:
        new_points = np.vstack(new_points)
    return new_points
    
def build_graph(detections, weight_source_sink=1,
                max_dist=0.07, max_jump=12, verbose=True,
                weight_confidence=weight_confidence,
                weight_distance=weight_distance):
    """
    Build the MOT graph. The edge weights are function of 
    the distance between detections.
    
    Parameters
    ----------
    detections : list of objects of type Detection or DetectionTracklet
        list of detections in any order
    weight_source_sink : float
        the weights of the edges connecting the source node 
        to all pre-nodes. The weight that connect the post-nodes 
        to the sink nodes are all set to 0.
    max_dist : float
        an edge between two detections is created if the distance separating them is less 
        than jump*max_dist where jump is the difference in time.
    max_jump : int
        maximum difference in time between two detections
    """
    assert isinstance(detections, (list, tuple))

    if len(detections)==0:
        return None
    
    if verbose:
        from tqdm import tqdm
        _tqdm = tqdm
    else:
        _tqdm = lambda x: x    
        
    is_with_tracklets = isinstance(detections[0], DetectionTracklet)
    is_with_detections = isinstance(detections[0], Detection)
    
    # sort the detection based on the index
    if is_with_tracklets:
        compare = lambda a,b: a.head.index-b.tail.index
        detections = sorted(detections, key=functools.cmp_to_key(compare))
    elif is_with_detections:
        compare = lambda a,b: a.index-b.index
        detections = sorted(detections, key=functools.cmp_to_key(compare))
    else:
        raise RuntimeError("Detection object must inherit from Detection or DetectionTracklet!")     

    # S->pre_i: represent how likely the detection i is the initial point 
    # pre_i->post_i: cost that reflects the reward of including the detection i
    # post_i->pre_j: encodes the similarity between detection i and j
    # post_i->T: represent how likely the detection i is the terminate point
    
    n_detections = len(detections)
    n_nodes = n_detections*2+2
    if verbose: 
        if is_with_detections:
            print("Number of frames: {}".format(len(set([d.index for d in detections]))))
            print("Number of detections: {}".format(n_detections))
        if is_with_tracklets:
            print("Number of tracklets: {}".format(n_detections))            

    SOURCE = 1
    SINK = n_nodes

    g = nx.DiGraph()
    g.add_node(SOURCE, label='source')
    g.add_node(SINK, label='sink')
    n = SOURCE+1

    # creates nodes for the detections, the edge going to source and sink and the pre->post edges
    m = 0
    for detection in detections:
        
        # S->pre-nodes
        g.add_node(n, detection=detection, label='pre-node', idet=m)
        if getattr(detection, "weight_source", None) is None:
            weight = weight_source_sink
        else:
            weight = detection.weight_source
        g.add_edge(SOURCE, n, weight=weight)
        detection.pre_node = n # save a reference

        # post-nodes->T
        n += 1
        g.add_node(n, detection=detection, label='post-node', idet=m)
        if getattr(detection, "weight_sink", None) is None:
            weight = 0   
        else:
            weight = detection.weight_sink
        g.add_edge(n, SINK, weight=weight)
        detection.post_node = n # save a reference       

        # pre-nodes->post-nodes
        g.add_edge(n-1, n, weight=weight_confidence(detection))
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

                        if is_with_tracklets:
                            jump = data2['detection'].tail.index-data1['detection'].head.index
                        elif is_with_detections:
                            jump = data2['detection'].index-data1['detection'].index
                        else:
                            raise RuntimeError("Detection object must inherit from Detection or DetectionTracklet!")      
                            
                        # create edges that go forward in time only
                        if jump>0 and jump<=max_jump:

                            if max_dist is not None:
                                if is_with_tracklets:
                                    dist = euclidean(data1['detection'].head.position, data2['detection'].tail.position)
                                elif is_with_detections:
                                    dist = euclidean(data1['detection'].position, data2['detection'].position)
                                else:
                                    raise RuntimeError("Detection object must inherit from Detection or DetectionTracklet!")
                            else:
                                dist = None
                            
                            if max_dist is None or dist<max_dist:
                                weight = weight_distance(jump, dist, data1['detection'], data2['detection'])
                                if weight is not None:
                                    n_post_pre += 1
                                    g.add_edge(n1, n2, weight=weight)   # post-nodes->pre-nodes

                        else:
                            # this speed things up big time
                            # for DetectionTracklets it's more complicated...detections overlap 
                            # As the graph with DetectionTracklets is in general very small we can scan it all 
                            if is_with_detections: 
                                if jump<=0:
                                    i_resume = i
                                else:
                                    # the nodes are ordered w.r.t the time so when we reach
                                    # this point all the nodes that follow can be discarded as well.
                                    # For this reason, we can break hear and continue. 
                                    break

    if verbose:
        print("Number of post-pre nodes edges created: {}".format(n_post_pre))
        
    if nx.has_path(g, SOURCE, SINK):
        return g
    else:
        return None

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
    
def _run_ssp(g, verbose=True, method='muSSP'):
    """
    Solve flow graph using Successive Shorthest Paths (SSP) method.
    Very fast
    """
    
    input_filename = '/tmp/graph.txt'
    output_filename = '/tmp/output.txt'
    
    save_graph(g, input_filename)
    
    curr_path = os.path.dirname(os.path.abspath(__file__))
    
    # solve graph
    cmd = [curr_path+"/{}/{}".format(method, method), "-i", input_filename, output_filename]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               universal_newlines=True)
    
    for line in process.stdout:
        if verbose:
            print(line.strip())
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

    # this recovers the trajectories from the output file. (can be optimized)
    with open(output_filename, "r") as f:
        res_edges = [ [ int(xx) for xx in x.strip().split(' ')[1:]] for x in list(f)]

    if len(res_edges):
        g_copy = g.copy()
        for s,t,r in res_edges:
            if r==0:
                g_copy.remove_edge(s,t)
        nodes = g.nodes()
        SOURCE = min(nodes)
        SINK = max(nodes)
        tracks_nodes = [ track[::2][1:] for track in list(nx.all_simple_paths(g_copy, SOURCE, SINK))]
    else:
        tracks_nodes = []
    
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
    trajectories = []
    n_discarded = 0
    for nodes in tracks_nodes:
        trajectory = [g.nodes[n]['detection'] for n in nodes]

        #if isinstance(trajectory[0], Detection):
        #    trajectory = interpolate_trajectory(trajectory)

        trajectories.append(trajectory)
    
    return trajectories

def plot_graph(graph, node_size=100, font_size=12, 
               node_color='y', edge_color='y', 
               linewidths=2,
               offset=np.array([0,0]), 
               source_pos=None, 
               target_pos=None, verbose=True, **kwargs):
    
    
    
    if verbose:
        if len(graph.nodes())>500:
            print("The graph is big. Plotting it may take a while.")
            
    positions = {}
    ps = []
    for n in graph.nodes():
        if 'detection' in graph.nodes[n]:
            if isinstance(graph.nodes[n]['detection'], DetectionTracklet):
                positions[n] = np.mean([d.position[:2] for d in graph.nodes[n]['detection'].tracklet], axis=0)
            elif isinstance(graph.nodes[n]['detection'], Detection):
                positions[n] = graph.nodes[n]['detection'].position[:2]
            else:
                print(n, graph.nodes[n])
                raise RuntimeError("Detection object must inherit from Detection or DetectionTracklet!") 
            ps.append(positions[n])   
    ps = np.array(ps)

    xmin, ymin = ps.min(0)
    xmax, ymax = ps.max(0)
    w,h = (xmax-xmin), (ymax-ymin)

    pos = {}
    for n in graph.nodes():
        node = graph.nodes[n]
        if node['label']=='source':
            p = source_pos if source_pos is not None else np.array([xmin-w*0.15, ymin-h*0.15])
        elif node['label']=='sink':
            p = target_pos if target_pos is not None else np.array([xmax+w*0.15, ymax+h*0.15])
        elif node['label']=='pre-node':
            p = positions[n]-np.array([w*0.025, 0])
        elif node['label']=='post-node':
            p = positions[n]+np.array([w*0.025, 0])
            
        pos[n] = p+offset
 
    nx.draw_networkx(graph, pos=pos, node_size=node_size, node_color=node_color,
                     edge_color=edge_color, font_size=font_size, **kwargs)
    #plt.gca().invert_yaxis()
    plt.legend()
    
def concat_tracklets(tracklets):
    '''
    Combine tracklets into a single trajectory
    Parameters:
    ----------
    tracklets : list of lists of objects of type Detection
    
    Returns:
    -------
    trajectory: list of object of type Detection
    '''
    return list(itertools.chain(*[d.tracklet for d in tracklets]))
    
def interpolate_(vector1, t1, vector2, t2):
    from scipy.interpolate import interp1d
    
    t_new = np.arange(t1+1, t2)
    f = interp1d(np.array([t1,t2]), np.array([vector1,vector2]).T)
    vector_new = f(t_new).T
    
    return vector_new, t_new   

def interpolate_trajectory(trajectory, features=False):
    """
    Fills the missing values in the trajectory using linear interpolation
    
    Note: this functions only interpolates missing detections.
    In other words only if there are jumps in the indexes.
    If you want to interpolate missing features you have to delete the detections first.
    """
    new_positions, new_indexes, new_features = None, None, None
    
    DetectionType = type(trajectory[0])

    new_trajectory = [trajectory[0]]
    for curr in trajectory[1:]:

        past = new_trajectory[-1]

        if curr.index-past.index==1:
            new_trajectory.append(curr)
        else:

            new_positions, new_indexes = interpolate_(past.position, past.index,
                                                      curr.position, curr.index)

            if features:
                new_features = {}
                for feature in features:
                    v1 = np.array(past.features[feature])
                    v2 = np.array(curr.features[feature])
                    shape = (-1,)+v1.shape
                    new_features[feature] = interpolate_(v1.ravel(), past.index,
                                                         v2.ravel(), curr.index)[0].reshape(shape)

            for i in range(len(new_indexes)):
                features_ = {}
                if features:
                    for feature in features:
                        features_[feature] = new_features[feature][i].tolist()
                        
                new_trajectory.append(DetectionType(index=int(new_indexes[i]), 
                                                    position=new_positions[i], 
                                                    features=features_,
                                                    confidence=None, 
                                                    datetime=None))

            new_trajectory.append(curr)
            
    return new_trajectory

def smooth_trajectory(trajectory, n=1, s=100, k=3):
    """
    Spline smoothing
    """
    trajectory_ = copy.deepcopy(trajectory)
    
    for i in range(n, len(trajectory_)-n-1):
        
        window = np.array([detection.position for detection in trajectory[i-n:i+n+1]])
        window += np.random.rand(*window.shape)/1e5
        
        tck, u = interpolate.splprep(window.T, s=s, k=k)

        new_position = np.column_stack(interpolate.splev(u[n], tck))[0]
        
        trajectory_[i].position = list(new_position)
    
    return trajectory_

def uniform_param(P):
    """
    Uniform parametrization for splines
    """
    u = np.linspace(0, 1, len(P))
    return u

def generate_param(P, alpha):
    n = len(P)
    u = np.zeros(n)
    u_sum = 0
    for i in range(1,n):
        u_sum += np.linalg.norm(P[i,:]-P[i-1,:])**alpha
        u[i] = u_sum
    
    return u/max(u)
    
def chordlength_param(P):
    """
    Chordlength parametrization for splines
    """
    u = generate_param(P, alpha=1.0)
    return u
    
def centripetal_param(P):
    """
    Centripetal parametrization for splines
    """
    u = generate_param(P, alpha=0.5)
    return u

def smooth_polynomial_2d(trajectory, degree=3):
    """
    Fits a trajectory with a polynomial function then 
    returns a new one with equal length segments.
    Work for 2D trajectories only.
    This function is suited for removing noise from short trajectories (tracklets).
    """
    positions = np.array([detection.position for detection in trajectory])
    
    # fit polynomial
    p = np.polyfit(positions[:,0], positions[:,1], degree)

    x = np.linspace(positions[0,0], positions[-1,0], len(positions))
    y = np.polyval(p, x)

    # get equal length segments
    u = uniform_param(positions)
    tck, ub = interpolate.splprep([x,y], s=0, k=2, u=u)

    u = np.linspace(0,1,len(positions))
    x2,y2 = interpolate.splev(u, tck)
    
    for detection,new_position in zip(trajectory,x2,y2):
        detection.position = new_position
    
    return trajectory
            
colors = [[255,0,0], [0,255,0], 
          [100,100,255], [255,255,0], 
          [0,255,255], [255,0,255],
          [225,225,225], [0,0,0],
          [128,128,128], [50,128,50]]+[np.random.randint(0,255,3).tolist() for _ in range(200)]    
    
def plot_trajectories(trajectories, axis=(0,1), linewidth=2, nodesize=7, 
                      display_time=False, fontsize=8, display_time_every=1, filter_index=None):
    import matplotlib.pyplot as plt
    plt.figure()
    for track,color in zip(trajectories, colors):
        color = tuple(c/255.0 for c in color)
        positions = []
        times = []
        for detection in track:
            
            if isinstance(detection, DetectionTracklet):
                positions_ = [[d.position[i] for i in axis] for d in detection.tracklet]
                time_ = [d.index for d in detection.tracklet]
            elif isinstance(detection, Detection):
                positions_ = [[detection.position[i] for i in axis]]
                time_ = [detection.index]
            else:
                raise RuntimeError("Detection object must inherit from Detection or DetectionTracklet!")            
            
            if filter_index is None:
                for p,t in zip(positions_, time_):
                    positions.append(p)
                    times.append(t)    
            else:
                for p,t in zip(positions_, time_):
                    if t >= filter_index[0] and t<filter_index[1]:
                        positions.append(p)
                        times.append(t) 

        if len(positions):
            positions = np.array(positions)[:,axis]
            times = np.array(times)
            plt.plot(positions[:,0], positions[:,1], '.-', color=color, linewidth=linewidth, markersize=nodesize)
            if display_time:
                for (x,y),time in zip(positions[::display_time_every], times[::display_time_every]):
                    plt.text(x,y, str(time), color=color, fontsize=fontsize, 
                             bbox={'facecolor': 'black', 'alpha': 0.8, 'pad': 1})
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
        cmd="ffmpeg -framerate {} -pattern_type glob -i '{}/*.{}' -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {} -vcodec h264 -y".format(fps, output_path, ext, output_video)
        print(cmd)
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
        for line in out.stdout:
            print(line.strip())   
    
def cost_trajectory(trajectory, weight_source=0, weight_sink=0,
                    weight_confidence=weight_confidence,
                    weight_distance=weight_distance):
    """
    Computes the cost of a trajectory
    
    Parameters
    ----------
    trajectory : list of object of type Detection
        sequence of detections
    weight_source : float
        weight of the edge linking the first detection to the source node
    weight_sink : float
        weight of the edge linking the last detection to the sink node
    """
    costs = [weight_source]
    for detection1, detection2 in zip(trajectory[:-1], trajectory[1:]):  
        jump = detection2.index-detection1.index
        dist = euclidean(detection1.position, detection2.position)
        
        costs.append(weight_confidence(detection1))
        costs.append(weight_distance(jump, dist, detection1, detection2))
        
    costs.append(weight_confidence(trajectory[-1]))    
    costs.append(weight_sink)
    
    return sum(costs)

def one_to_one_matching(detections1, detections2, weight_source_sink=0.0,
                        max_dist=0.07, max_jump=12, verbose=True,
                        weight_confidence=weight_confidence,
                        weight_distance=weight_distance):
    
    if len(detections1)==0 or len(detections2)==0:
        return [None]*len(detections2), [None]*len(detections2)
    
    detections1_ = copy.deepcopy(detections1)
    detections2_ = copy.deepcopy(detections2)
                        
    def weight_distance_mod(jump, distance, detection1, detection2):
        jump = np.abs(detection2.index_orig-detection1.index_orig)
        if jump>max_jump:
            return None # since we have forced the indexes to be 0 or 1 we handle the >max_jump statement here
        return weight_distance(jump, distance, detection1, detection2)
    
    detections_ssp = []
    for i, d in enumerate(detections1_):
        if isinstance(d, DetectionTracklet):
            d.index_orig = d.head.index
            d.head.index = 0 # we force the index to be 0 and 1 so that detections remain in the correct order
        else:
            d.index_orig = d.index
            d.index = 0 # we force the index to be 0 and 1 so that detections remain in the correct order
        d.weight_sink = 9999
        d.i = i
        detections_ssp.append(d)
    for j, d in enumerate(detections2_):
        if isinstance(d, DetectionTracklet):
            d.index_orig = d.tail.index
            d.tail.index = 1 # we force the index to be 0 and 1 so that detections remain in the correct order
        else:
            d.index_orig = d.index
            d.index = 1 # we force the index to be 0 and 1 so that detections remain in the correct order
        d.weight_source = 9999
        d.i = j
        detections_ssp.append(d)

    g = build_graph(detections_ssp, 
                    weight_source_sink=weight_source_sink,
                    max_dist=max_dist, 
                    max_jump=1, 
                    verbose=verbose, 
                    weight_distance=weight_distance_mod,
                    weight_confidence=weight_confidence)
    if g is None:
        trajectories = [] 
    else:
        trajectories = solve_graph(g, verbose=verbose)   
                 
    matches1 = [None]*len(detections1)
    matches2 = [None]*len(detections2)
    for trajectory in trajectories:
        if len(trajectory)==1:
            continue
        elif len(trajectory)>2:
            raise RuntimeError("resulting trjectory contains more then two detections!")
        matches1[trajectory[0].i] = trajectory[1].i
        matches2[trajectory[1].i] = trajectory[0].i
                        
    return matches1, matches2