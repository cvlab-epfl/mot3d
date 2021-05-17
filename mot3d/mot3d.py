import numpy as np
import networkx as nx
import time
import copy
import functools

from .types import *
from .weight_functions import weight_distance_detections_2d, weight_confidence_detections_2d
from .utils.trajectory import concat_tracklets

from .utils.utils import pickle_write

__all__ = ["build_graph", "solve_graph", "graph_to_text", "save_graph"]
    
def build_graph(detections, weight_source_sink=1,
                max_jump=12, verbose=True,
                weight_confidence=weight_confidence_detections_2d,
                weight_distance=weight_distance_detections_2d):
    """
    Build the flow graph.
    
    Parameters
    ----------
    detections : list of objects of type Detection or DetectionTracklet
        the detections in the list can be in any order
    weight_source_sink : float
        This value defines the threshold, in therm of path cost, for the creation of a trajectory.
        The weight of the edges connecting the source node to the pre-nodes are set to this value. 
        The weight of the edges connecting the post-nodes to the sink nodes are all set to 0.
    max_jump : int
        maximum difference in time between two detections
    weight_confidence : function
        Function defining the weight for the pre-node->post-node edges.
    weight_distance : function
        Function defining the weight for the post-node->pre-node edges.
        
    Return
    ------
    Flow graph (networkx DiGraph)
    
    """
    assert isinstance(detections, (list, tuple))

    if len(detections)==0:
        return None
    
    if verbose:
        from tqdm import tqdm
        _tqdm = tqdm
    else:
        _tqdm = lambda x: x    
    
    # sort the detections based on the index
    compare = lambda a,b: -a.diff_index(b)
    detections = sorted(detections, key=functools.cmp_to_key(compare))   
    
    is_with_tracklets = isinstance(detections[0], DetectionTracklet)
    is_with_detections = isinstance(detections[0], Detection)    
    
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

    # creates two nodes for each detections and the edges connecting source and sink and the pre->post edges
    m = 0
    for detection in detections:
        
        # S->pre-nodes
        g.add_node(n, detection=detection, label='pre-node', idet=m)
        detection.pre_node = n # save a reference
        if detection.entry_points:
            g.add_edge(SOURCE, n, weight=weight_source_sink)
        
        n += 1

        # post-nodes->T
        g.add_node(n, detection=detection, label='post-node', idet=m)
        detection.post_node = n # save a reference
        if detection.exit_point:            
            g.add_edge(n, SINK, weight=0)

        # pre-nodes->post-nodes
        g.add_edge(n-1, n, weight=weight_confidence(detection))
        n += 1
        m += 1

    # creates the post-node->pre-node edges 
    i_resume = 0
    ndata = list(enumerate(g.nodes(data=True)))

    n_post_pre = 0
    for _,(n1,data1) in _tqdm(ndata):
        if data1['label']=='post-node':

            for i,(n2,data2) in ndata[i_resume:]:
                if data2['label']=='pre-node':

                    if data1['idet']!=data2['idet']:
                        
                        jump = data1['detection'].diff_index(data2['detection'])
                        
                        # create edges that go forward in time only
                        if jump>0 and jump<=max_jump:

                            weight = weight_distance(data1['detection'], data2['detection'])
                            if weight is not None:
                                n_post_pre += 1
                                g.add_edge(n1, n2, weight=weight) # post-nodes->pre-nodes                            

                        else:
                            # since the detections are sorted by index, when jump<=0 we can
                            # save the current index of this for loo and then in the next iteration start back from it.
                            # This reduces the time required to build the graph big times.
                            # For DetectionTracklets cannot be done because tracklets cna overlap so we simply skip this.
                            
                            if is_with_detections: 
                                if jump<=0:
                                    i_resume = i
                                else:
                                    break

    if verbose:
        print("Number of post-pre nodes edges created: {}".format(n_post_pre))
        
    if nx.has_path(g, SOURCE, SINK):
        return g
    else:
        return None    
    
def _run_ssp(g, verbose=1, method='muSSP'):
    """
    Solve the flow graph using Successive Shorthest Paths (SSP) method.
    Note: Very fast
    """
    graph_as_text = graph_to_text(g)
    
    if method=='muSSP':
        from .solvers.wrappers.muSSP import wrapper 
        edges = wrapper.solve(graph_as_text, int(verbose))
    else:
        raise NotImplementedError(method)

    g_sub = g.edge_subgraph(edges)
    SOURCE = min(g_sub.nodes())
    SINK = max(g_sub.nodes())
    paths = [ track[::2][1:] for track in list(nx.all_simple_paths(g_sub, SOURCE, SINK))]
    
    return paths

def _run_ilp(g, verbose=True):
    """
    Solve the flow graph using Integer Lienar Programming method.
    Note: Slow
    """
    from .solvers.ilp import ilp
    
    tracks_nodes = ilp.solve_graph_ilp(g, verbose)
    
    return tracks_nodes

def solve_graph(g, verbose=True, method='muSSP'):
    
    start_time = time.time()
    
    if method in ['muSSP', 'FollowMe', 'SSP']:
        tracks_nodes = _run_ssp(g, verbose, method)
    elif method == 'ILP':
        tracks_nodes = _run_ilp(g, verbose)
    else:
        raise ValueError("Unrecognazed method '{}'. Choose 'muSSP' or 'ILP'".format(method))
        
    trajectories = []
    if len(tracks_nodes):
        
        is_with_tracklets = isinstance(g.nodes[tracks_nodes[0][0]]['detection'], DetectionTracklet)

        for nodes in tracks_nodes:
            if is_with_tracklets:
                tracklets = [g.nodes[n]['detection'].tracklet for n in nodes]
                trajectory = concat_tracklets(tracklets)
                trajectories.append(trajectory)
            else:
                trajectory = [g.nodes[n]['detection'] for n in nodes]
                trajectories.append(trajectory)
    
    if verbose:
        print("Graph solved in {:0.4f}s using {}.".format(time.time()-start_time, method))
    
    return trajectories

def graph_to_text(g):
    
    graph_as_text = [] 
    graph_as_text.append("p min {} {}".format(len(g), len(g.edges())))

    graph_as_text.append("c ------ source->pre-nodes ------")
    for s,t,data in g.edges(data=True):
        if g.nodes[s]['label']=='source' and g.nodes[t]['label']=='pre-node':
            graph_as_text.append("a {} {} {:0.7f}".format(s, t, data['weight']))

    graph_as_text.append("c ------ post-nodes->sink ------")  
    for s,t,data in g.edges(data=True):
        if g.nodes[s]['label']=='post-node' and g.nodes[t]['label']=='sink':
            graph_as_text.append("a {} {} {:0.7f}".format(s, t, data['weight']))      

    graph_as_text.append("c ------ pre-node->post-nodes ------")
    for s,t,data in g.edges(data=True):
        if g.nodes[s]['label']=='pre-node' and g.nodes[t]['label']=='post-node':
            graph_as_text.append("a {} {} {:0.7f}".format(s, t, data['weight']))

    graph_as_text.append("c ------ post-node->pre-nodes ------") 
    for s,t,data in g.edges(data=True):
        if g.nodes[s]['label']=='post-node' and g.nodes[t]['label']=='pre-node':
            graph_as_text.append("a {} {} {:0.7f}".format(s, t, data['weight']))   

    return graph_as_text
    
def save_graph(graph_as_text, filename="/tmp/graph.txt"):
    
    fd = os.open(filename, os.O_RDWR|os.O_CREAT|os.O_SYNC|os.O_TRUNC) 
    
    for string in graph_as_text:
        os.write(fd, str.encode(string+"\n"))    

    os.close(fd)