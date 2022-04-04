#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# ---------------------------------------------------------------------------
import sys
import os
import numpy as np
import networkx as nx
import imageio
import subprocess
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from .utils import mkdir
from ..types import Detection, DetectionTracklet

colors = [[255,0,0], [0,255,0], 
          [100,100,255], [255,255,0], 
          [0,255,255], [255,0,255],
          [225,225,225], [0,0,0],
          [128,128,128], [50,128,50]]+[np.random.randint(0,255,3).tolist() for _ in range(1000)]  

def draw_points(image, centers, radius, color='r'): 
    """ Draws filled point on the image
    """
    _image = image.copy()        
    if color=='r':
        color = [255,0,0]
    elif color=='g':
        color = [0,255,0]
    elif color=='b':
        color = [0,0,255]
    elif color=='w':
        color = [255,255,255]
    elif color=='k':
        color = [0,0,0]
    
    for point in centers:
        _image = cv2.circle(_image, tuple(point.astype(np.int)), radius, color=color, thickness=-1)
    return _image

def draw_rectangles(image, centers, size, color='r', thickness=3): 
    """ Draws rectangles on the image
    """ 
    _image = image.copy()
    if color=='r':
        color = [255,0,0]
    elif color=='g':
        color = [0,255,0]
    elif color=='b':
        color = [0,0,255]
    elif color=='w':
        color = [255,255,255]
    elif color=='k':
        color = [0,0,0]
        
    for i, (x,y) in enumerate(np.int_(centers)):
        pt1 = (x-size[1]//2, y-size[0]//2)
        pt2 = (x+size[1]//2, y+size[0]//2)
        _image = cv2.rectangle(_image, pt1, pt2, color=color, thickness=thickness)
    return _image

def plot_graph(graph, node_size=100, font_size=12, 
               node_color='y', edge_color='y', 
               linewidths=2,
               offset=np.array([0,0]), 
               source_pos=None, 
               target_pos=None, 
               show_source_sink_nodes=True,
               verbose=True, **kwargs):
    
    graph_ = graph.copy()
    
    if verbose:
        if len(graph_.nodes())>500:
            print("The graph is big. Plotting it may take a while.")
            
    positions = {}
    ps = []
    for n in graph_.nodes():
        if 'detection' in graph_.nodes[n]:
            if isinstance(graph_.nodes[n]['detection'], DetectionTracklet):
                positions[n] = np.mean([d.position[:2] for d in graph_.nodes[n]['detection'].tracklet], axis=0)
            elif isinstance(graph_.nodes[n]['detection'], Detection):
                positions[n] = graph_.nodes[n]['detection'].position[:2]
            else:
                raise RuntimeError("Detection object must inherit from Detection or DetectionTracklet not '{}'!".format(type(graph_.nodes[n]['detection']))) 
            ps.append(positions[n])   
    ps = np.array(ps)

    xmin, ymin = ps.min(0)
    xmax, ymax = ps.max(0)
    w,h = (xmax-xmin), (ymax-ymin)
    
    nodes = graph_.nodes()
    SOURCE = min(nodes)
    SINK = max(nodes)   
    
    if not show_source_sink_nodes:
        graph_.remove_node(SOURCE)
        graph_.remove_node(SINK)

    pos = {}
    for n in graph_.nodes():
        node = graph_.nodes[n]
        if node['label']=='source':
            p = source_pos if source_pos is not None else np.array([xmin-w*0.15, ymin-h*0.15])
        elif node['label']=='sink':
            p = target_pos if target_pos is not None else np.array([xmax+w*0.15, ymax+h*0.15])               
        elif node['label']=='pre-node':
            p = positions[n]-np.array([w*0.025, 0])
        elif node['label']=='post-node':
            p = positions[n]+np.array([w*0.025, 0])
            
        pos[n] = p+offset
 
    nx.draw_networkx(graph_, pos=pos, node_size=node_size, node_color=node_color,
                     edge_color=edge_color, font_size=font_size, **kwargs)
    #plt.gca().invert_yaxis()
    plt.legend()  
    
def plot_trajectories(trajectories, axis=(0,1), linewidth=2, nodesize=7, 
                      display_time=False, fontsize=8, display_time_every=1, 
                      filter_index=None, calibration=None):
    import matplotlib.pyplot as plt

    for track,color in zip(trajectories, colors):
        color = tuple(c/255.0 for c in color)
        positions = []
        times = []
        for detection in track:
            
            if isinstance(detection, DetectionTracklet):
                positions_ = [d.position for d in detection.tracklet]
                time_ = [d.index for d in detection.tracklet]
            elif isinstance(detection, Detection):
                positions_ = [detection.position]
                time_ = [detection.index]
            else:
                raise RuntimeError("Detection object must inherit from Detection or DetectionTracklet not '{}'!".format(type(detection)))            
            
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
            positions = np.array(positions)
            if calibration is not None and positions.shape[1]==3:
                K = np.array(calibration['K'])
                R = np.array(calibration['R'])
                t = np.array(calibration['t'])
                dist = np.array(calibration['dist'])
                positions = cv2.projectPoints(positions, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1,2)  
            else:
                positions = positions[:,axis]
            times = np.array(times)
            plt.plot(positions[:,0], positions[:,1], '.-', color=color, linewidth=linewidth, markersize=nodesize)
            if display_time:
                for (x,y),time in zip(positions[::display_time_every], times[::display_time_every]):
                    plt.text(x,y, str(time), color=color, fontsize=fontsize, 
                             bbox={'facecolor': 'grey', 'alpha': 0.8, 'pad': 1})
                # making sure we add the last one too
                for (x,y),time in zip(positions[[-1]], times[[-1]]):
                    plt.text(x,y, str(time), color=color, fontsize=fontsize, 
                             bbox={'facecolor': 'grey', 'alpha': 0.8, 'pad': 1})
    plt.grid()
        
def visualisation(filenames, tracks, indexes, calibration=None, bboxes=None, 
                  crop=(slice(None,None), slice(None,None)), trace_length=25, thickness=5, thickness_boxes=2,
                  output_path="./output/sequence1", output_video="sequence1.mp4", fps=25,
                  img_preprocessing=None):
    
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
    
    if crop is None:
        crop=(slice(None,None), slice(None,None))

    mkdir(output_path) 

    for i,filename in tqdm(enumerate(filenames)):

        basename = os.path.basename(filename)
        img = imageio.imread(filename)
        
        if img_preprocessing is not None:
            img = img_preprocessing(img, i)

        for j,(track,index,color) in enumerate(zip(tracks, indexes, colors)):
            
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
                    if bboxes[j][ii] is not None:
                        xmin, ymin, xmax, ymax = bboxes[j][ii]
                        img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=color, thickness=thickness_boxes)

        imageio.imwrite(os.path.join(output_path, basename), img[crop]) 
        
    if isinstance(output_video, str):
        ext = basename.split('.')[-1]
        cmd="ffmpeg -framerate {} -pattern_type glob -i '{}/*.{}' -pix_fmt yuv420p -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -movflags +faststart -preset slow -profile:v baseline -vcodec h264 {} -y".format(fps, output_path, ext, output_video)
        print(cmd)
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
        for line in out.stdout:
            print(line.strip())  