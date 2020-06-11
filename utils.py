import os
import sys
import json
import re
import os
import ast
import glob
import pickle
import numpy as np
import cv2

__all__ = ["json_read", "json_write", "pickle_read", "pickle_write", 
           "mkdir", "sort_nicely", "find_files", "create_unet_labels",
           "nonmaxima_suppression", "nonmaxima_suppression_fast", 
           "undistort_points", "invert_Rt", "triangulate", "project_points",
           "draw_points", "draw_rectangles", "create_unet_labels_gradient"]

def json_read(filename):
    try:
        with open(os.path.abspath(filename)) as f:    
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))
        
def json_write(filename, data):
    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        raise ValueError("Unable to write JSON {}".format(filename))   
        
def pickle_read(filename):
    with open(filename, "rb") as f:    
        data = pickle.load(f)
    return data

def pickle_write(filename, data):
    directory = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)        

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None, recursive=False):
    # make sure to use ** in file_or_folder when using recusive
    # ie find_files("folder/**", "*.json", recursive=True)
    import os
    import glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder, recursive=recursive)]
    filenames = sort_nicely(filenames)    
    filename_files = []
    for filename in filenames:
        if os.path.isfile(filename):
            filename_files.append(filename)                 
    return filename_files

def dict_keys_to_string(d):
    return {str(key):value for key,value in d.items()}

def dict_keys_from_literal_string(d):
    new_d = {}
    for key,value in d.items():
        if isinstance(key, str):
            try:
                new_key = ast.literal_eval(key)
            except:
                new_key = key
        else:
            new_key = key
        new_d[new_key] = value
    return new_d

def find_closest(point, points): 
    dists = np.linalg.norm(points-point[None], axis=1)
    idx_min = np.argmin(dists)
    dist_min = dists[idx_min]
    return dist_min, idx_min

def invert_Rt(R, t):
    Ri = R.T
    ti = np.dot(-Ri, t)
    return Ri, ti

def project_points(pts, K, R, t, dist=None):
    pts_ = np.reshape(pts, (-1,3))
    rvec = cv2.Rodrigues(R)[0]
    proj = cv2.projectPoints(pts_, rvec, t, K, dist)[0].reshape(-1,2)
    return proj

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

def hungarian_matching(pos1, pos2, radius_match=0.3):
    from munkres import Munkres
    
    pos1 = np.reshape(pos1, (-1,2))
    pos2 = np.reshape(pos2, (-1,2))   

    n1 = pos1.shape[0]    
    n2 = pos2.shape[0]

    n_max = max(n1, n2)
    
    if n_max==0:
        return None, None, [], [], []

    # building the cost matrix based on the distance between 
    # detections and ground-truth positions
    matrix = np.ones((n_max, n_max))*9999999
    for i in range(n1):    
        for j in range(n2):

            d = np.sqrt(((pos2[j,0] - pos1[i,0])**2 + \
                         (pos2[j,1] - pos1[i,1])**2))

            if d <= radius_match:
                matrix[i,j] = d

    indexes = Munkres().compute(matrix.copy())

    TP = []   
    matched1 = np.zeros(len(pos1), np.bool)
    matched2 = np.zeros(len(pos2), np.bool)
    for i, j in indexes:
        value = matrix[i][j]
        if value <= radius_match:
            TP.append(j)
            matched1[i] = True
            matched2[j] = True
        else:
            TP.append(None)
            
    FP = np.where(matched2==False)[0].tolist()
    FN = np.where(matched1==False)[0].tolist()

    return TP, FP, FN