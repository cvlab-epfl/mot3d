import numpy as np
import itertools
from scipy import interpolate
from collections import defaultdict
import copy

def concat_tracklets(tracklets):
    return list(itertools.chain(*tracklets))
    
def interpolate_(vector1, t1, vector2, t2):
    
    t_new = np.arange(t1+1, t2)
    f = interpolate.interp1d(np.array([t1,t2]), np.array([vector1,vector2]).T)
    vector_new = f(t_new).T
    
    return vector_new  

def interpolate_trajectory(trajectory, attr_names=['position']):
    """
    Fills the missing detections in the trajectory using linear interpolation
    
    Note: this functions does not modify the existing detections! 
    It only adds new ones to fill the holes.
    
    If you have other attributes to interpolates such as 'bbox' or 'color_histogram'
    simply add them to attr_names.
    """
    new_positions, new_indexes, new_features = None, None, None
    
    DetectionType = type(trajectory[0])

    new_trajectory = [trajectory[0]]
    for curr in trajectory[1:]:

        past = new_trajectory[-1]

        if curr.index-past.index==1:
            new_trajectory.append(curr)
        else:
            
            new_indexes = np.arange(past.index+1, curr.index)

            new_attrs = {}
            for name in attr_names:
                
                x1 = getattr(past, name, None)
                x2 = getattr(curr, name, None)
                
                if x1 is not None and x2 is not None:
                    x1 = np.asarray(x1)
                    x2 = np.asarray(x2)
                    new_attrs[name] = interpolate_(np.ravel(x1), past.index,
                                                   np.ravel(x2), curr.index)
                    new_attrs[name] = new_attrs[name].reshape((-1,)+x1.shape)
                else:
                    raise ValueError("Detection does not have attribute {}!".format(name))

            for i,index in enumerate(new_indexes):
                detection = DetectionType(index=index, 
                                          **{name:attr[i] for name, attr in new_attrs.items()})
                new_trajectory.append(detection)

            new_trajectory.append(curr)
            
    return new_trajectory

def smooth_trajectory(trajectory, s=None, k=2, attr_names=['position']):
    """
    This function fits a spline to each class attribute then apply a smoothing on them.
    
    Parameters
    ----------
    trajectory : list of object of type Detection
        the type Detection can have different attributes such as postion, bbox, color_histogram
    s : int
        amount of smoothing to apply. Usual range 10..10000.
        If None "automatic".
    k: int
        degree of the spline to fit
    attr_names : list of strings
        the class attribute to smooth
    """
    
    indexes = []
    data = defaultdict(lambda: [])
    for detection in trajectory:
        indexes.append(detection.index)
        for name in attr_names:
            data[name].append(getattr(detection, name))
    
    data_smooth = {}
    for name in attr_names:
        
        # remove None if presents
        _indexes, _x = list(zip(*[(i,x) for i,x in zip(indexes, data[name]) if x is not None]))
        
        # reshape to what splprep expetcs
        _x = np.array(_x)
        shape = _x.shape
        _x = _x.reshape(_x.shape[0], -1)
        
        # removing duplicates
        _,idxs = np.unique(_x, return_index=True, axis=0)  
        _indexes = np.array(_indexes)[idxs]
        _x = _x[idxs]
        idxs2 = np.argsort(_indexes)
        _indexes = _indexes[idxs2]
        _x = _x[idxs2]    
        
        # interpolate + smoothing
        try:
            tck, _ = interpolate.splprep(_x.T, s=s, k=k, u=_indexes)
        except:
            print(_x, _indexes, s, k)
            raise
        _x_new = np.array(interpolate.splev(indexes, tck)).T
        _x_new = _x_new.reshape(*shape)
        
        data_smooth[name] = _x_new.tolist()
    
    trajectory_ = copy.deepcopy(trajectory) 
    for i,index in enumerate(indexes):
        for name in attr_names:
            setattr(trajectory_[i], name, data_smooth[name][i])
    
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

def split_trajectory_at_indexes(trajectory, at_indexes=[10, 20, 30]):

    idxs = [d.index for d in trajectory]
    
    def split(traj, idx):
        i = 0
        for d in traj:
            if d.index>idx:
                break
            i += 1
        return traj[:i], traj[i:]
    
    remaining = trajectory
    
    tracklets = []
    for idx in sorted(at_indexes):
        tracklet, remaining = split(remaining, idx)
        if len(tracklet):
            tracklets.append(tracklet)
        if not len(remaining):
            break
    if len(remaining):
        tracklets.append(remaining)
        
    return tracklets

def split_trajectory_modulo(trajectory, length=10):

    idxs = [d.index for d in trajectory]
    
    def split(traj):
        i = 0
        prev_modulo = -1
        for d in traj:
            curr_modulo = d.index%length
            if curr_modulo < prev_modulo:
                break
            prev_modulo = curr_modulo
            i += 1
        return traj[:i], traj[i:]
    
    remaining = trajectory
    
    tracklets = []
    while len(remaining):
        tracklet, remaining = split(remaining)
        if len(tracklet):
            tracklets.append(tracklet)
    if len(remaining):
        tracklets.append(remaining)
        
    return tracklets

def remove_short_trajectories(trajectories, th_length=10, verbose=False):
    new_trajectories = []
    for traj in trajectories:
        if len(traj)>th_length:
            new_trajectories.append(traj)
    return new_trajectories

def bboxes_overlap_matrix(bboxes1, bboxes2):
    
    bboxes1_ = np.array(bboxes1)
    bboxes2_ = np.array(bboxes2)

    x1 = np.max([bboxes1_[:,0], bboxes2_[:,0]], axis=0)
    y1 = np.max([bboxes1_[:,1], bboxes2_[:,1]], axis=0)
    x2 = np.min([bboxes1_[:,2], bboxes2_[:,2]], axis=0)
    y2 = np.min([bboxes1_[:,3], bboxes2_[:,3]], axis=0)

    inter_area = np.abs(np.max([x2 - x1, np.zeros(len(x1))], axis=0) * np.max([y2 - y1, np.zeros(len(x1))], axis=0))

    bboxes1_area = np.abs((bboxes1_[:,2] - bboxes1_[:,0]) * (bboxes1_[:,3] - bboxes1_[:,1]))
    bboxes2_area = np.abs((bboxes2_[:,2] - bboxes2_[:,0]) * (bboxes2_[:,3] - bboxes2_[:,1]))

    p1 = inter_area / bboxes1_area
    p2 = inter_area / bboxes2_area

    return p1, p2

def trajectory_overlap_2d_bbox(trajectory1, trajectory2):
    
    # return if the two trajectories don't intersect
    if (trajectory2[0].index>trajectory1[-1].index) or \
        (trajectory1[0].index>trajectory2[-1].index):
        return 0, 0, 0, 0, []

    trajectory1_ = mot3d.interpolate_trajectory(trajectory1, features=['bbox'])
    trajectory2_ = mot3d.interpolate_trajectory(trajectory2, features=['bbox'])
    
    indexes1 = np.array([d.index for d in trajectory1_])
    indexes2 = np.array([d.index for d in trajectory2_])
    
    val, idxs1, idxs2 = np.intersect1d(indexes1, indexes2, assume_unique=True, return_indices=True)
    
    bboxes1 = np.array([trajectory1_[i].features['bbox'] for i in idxs1])
    bboxes2 = np.array([trajectory2_[j].features['bbox'] for j in idxs2])
        
    overlap12, overlap21 = bboxes_overlap_matrix(bboxes1, bboxes2)
    
    o12 = np.sum(overlap12)
    o21 = np.sum(overlap21)    
    
    s12 = o12/len(trajectory1_)
    s21 = o21/len(trajectory2_)  
    
    s12_w = o12/len(overlap12)
    s21_w = o21/len(overlap21)     
    
    overlap12_time = len(idxs1)/len(trajectory1_)
    overlap21_time = len(idxs2)/len(trajectory2_)
    
    return s12_w, s21_w, overlap12_time, overlap21_time, indexes1[idxs1].tolist()






# Given three colinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r): 
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
        (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False

def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 

    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 

    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 

        # Clockwise orientation 
        return 1
    elif (val < 0): 

        # Counterclockwise orientation 
        return 2
    else: 

        # Colinear orientation 
        return 0

# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 

    # Find the 4 orientations required for 
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 

    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True

    # Special Cases 

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True

    # If none of the cases 
    return False

def find_intersections_fast(trajectory1, trajectory2, window=5, step=5):
    """ Finds the approximate position of all intersections along the two trajectoires.
    It does not find intersection if ...
    """
    assert step<=window
    
    # return if the two trajectories don't intersect at all
    if (trajectory2[0].index>trajectory1[-1].index) or\
        (trajectory1[0].index>trajectory2[-1].index):
        return False

    # the range of indexes that intersect 
    intersection = (max(trajectory1[0].index, trajectory2[0].index),
                    min(trajectory1[-1].index, trajectory2[-1].index))

    # prepare a pointer for each trajectory poiting to 
    # the first element of the intersection
    i = [0,0] # [start,end] of the window
    while trajectory1[i[0]].index<=(intersection[0]-window): 
        i[0] += 1
    i[1] = i[0]
    while i[1]<(len(trajectory1)-1) and trajectory1[i[1]].index<(intersection[0]+window): 
        i[1] += 1       

    j = [0,0] # [start,end] of the window
    while trajectory2[j[0]].index<=(intersection[0]-window): 
        j[0] += 1
    j[1] = j[0]
    while j[1]<(len(trajectory2)-1) and trajectory2[j[1]].index<(intersection[0]+window): 
        j[1] += 1 

    index = intersection[0]

    intersections = []
    while i[1]<(len(trajectory1)-1) and j[1]<(len(trajectory2)-1):

        res = doIntersect(trajectory1[i[0]].position, trajectory1[i[1]].position,
                          trajectory2[j[0]].position, trajectory2[j[1]].position)
        if res:
            intersections.append((list(i),
                                  list(j),
                                 (trajectory1[i[0]].index, trajectory1[i[1]].index),
                                 (trajectory2[j[0]].index, trajectory2[j[1]].index)))

        index += step

        while i[1]<(len(trajectory1)-1) and trajectory1[i[1]].index<(index+window): 
            i[1] += 1
        while trajectory1[i[0]].index<(index-window): 
            i[0] += 1
        while j[1]<(len(trajectory2)-1) and trajectory2[j[1]].index<(index+window):    
            j[1] += 1            
        while trajectory2[j[0]].index<(index-window): 
            j[0] += 1
        
    if len(intersections)<=1:
        return intersections
    
    # merge ranges that intersect
    agglos = []
    agglo = [0]
    for i in range(1,len(intersections)):
        if intersections[i-1][0][1]>=intersections[i][0][0]:
            agglo.append(i)
        else:
            agglos.append(agglo)
            agglo = [i]
    agglos.append(agglo)    
    
    intersections_merged = []
    for idxs in agglos:
        i = (intersections[idxs[0]][0][0], intersections[idxs[-1]][0][1])
        j = (intersections[idxs[0]][1][0], intersections[idxs[-1]][1][1])
        indexes1 = (intersections[idxs[0]][2][0], intersections[idxs[-1]][2][1])
        indexes2 = (intersections[idxs[0]][3][0], intersections[idxs[-1]][3][1])
        intersections_merged.append((i,j,indexes1, indexes2))
            
    return intersections_merged


def find_close_detections(traj1, traj2, index_window=3, dist=1):
    """
    Finds the elements in the two trajectories that are close in both index/time and space
    
    Parameters
    ----------
    traj1 & traj2 : list of objects of type Detection
    index_window : int
        time/index window for which an element in traj1 and traj2 are considered close
    dist : float
        max spatial distance for which an element in traj1 and traj2 are considered close 
        
    Return
    ------
    s1 : pecentage of close elements w.r.t length of traj1
    s2 : pecentage of close elements w.r.t length of traj2
    i : indexes close elements of traj1
    j : indexes close elements of traj2
    """
    from scipy.spatial.distance import cdist
    
    # return if the two trajectories don't intersect at all in time/index
    if (traj2[0].index>traj1[-1].index) or (traj1[0].index>traj2[-1].index):
        return 0, 0, [], [] 
    
    positions1 = np.array([list(d.position) for d in traj1])
    idxs_time1 = np.array([d.index for d in traj1])

    positions2 = np.array([list(d.position) for d in traj2])
    idxs_time2 = np.array([d.index for d in traj2])

    D_matrix = cdist(positions1, positions2)
    I_matrix = cdist(idxs_time1[:,None], idxs_time2[:,None])

    D_matrix[I_matrix>index_window] = np.inf
    i,j = np.where(D_matrix<dist)
    i,j = list(set(i)), list(set(j))

    n_close = len(i)
    s1 = n_close/len(traj1)
    s2 = n_close/len(traj2)
    
    return s1, s2, i, j

def split_trajectory(trajectory, indexes, boxfilter=None, verbose=False):
    """
    Split a trajectory by deleting elements. 
    The indexes of the element to delete can be in any order and can be consecutives. 
    The box filter can be used to unform a split for quasi-consequtive indexes.
    
    Note: the indexes here refers to the integer 
          values used to access array elements not the indexes/time of the detections.
          
    Parameters
    ----------
    trajectory : list of objects of type Detection
    indexes : list of integers ranges [0,len(trajectory)[
        can be in any order and with consecutive values
    boxfilter : int, optional
        if enabled a box filter of size boxfilter is applied to the mask/indexes
        
    Return
    ------
    trajectory_splits : new set of trajectories
    removed_trajectory_splits : the trajectories that have been cut out  
    
    Example
    -------
    >> trajectory = range(15)
    >> split_trajectory(trajectory, [6,8,10])
    >> split_trajectory(trajectory, [6,8,10], boxfilter=3)
    >> [[0, 1, 2, 3, 4, 5], [7], [9], [11, 12, 13, 14]]
    >> [[0, 1, 2, 3, 4], [12, 13, 14]]
    """
    if len(indexes)==0:
        return [trajectory],[]
    
    mask = np.ones(len(trajectory))
    mask[np.int_(indexes)] = 0
    
    if boxfilter:
        mask_filt = np.logical_not(np.convolve(np.logical_not(mask), np.ones(boxfilter)/boxfilter, 'same'))
        r = boxfilter//2
        mask = np.concatenate([mask[:r], mask_filt[r:-r], mask[-r:]])
        
    trajectory_splits = []
    new_traj = []
    
    removed_trajectory_splits = []
    removed_traj = []
    
    for d,m in zip(trajectory, mask):

        if m>=0.5:
            new_traj.append(d)
            if len(removed_traj):
                removed_trajectory_splits.append(removed_traj)
                removed_traj = []
        else:
            removed_traj.append(d)
            if len(new_traj):
                trajectory_splits.append(new_traj)
                new_traj = []
            
    if len(new_traj):
        trajectory_splits.append(new_traj)
    if len(removed_traj):
        removed_trajectory_splits.append(removed_traj)        
        
    return trajectory_splits, removed_trajectory_splits

def split_close_trajectories(trajectories, index_window=3, dist=20, boxfilter=4, verbose=False):

    new_trajectories = []
    to_split = {i:[] for i in range(len(trajectories))}
    for i, j in itertools.combinations(range(len(trajectories)), 2):
        traj1 = trajectories[i]
        traj2 = trajectories[j] 
        
        # skip if the two trajectories don't intersect at all in time/index
        if (traj2[0].index>traj1[-1].index) or (traj1[0].index>traj2[-1].index):
            continue

        _, _, idxs1, idxs2 = find_close_detections(traj1, traj2, index_window=index_window, dist=dist)
        if len(idxs1):
            to_split[i] += list(idxs1)
            to_split[j] += list(idxs2)    

    for i,idxs in to_split.items():
        new_trajectories += split_trajectory(trajectories[i], idxs, boxfilter=boxfilter)[0]
        
    return new_trajectories

def split_intersecting_trajectories(trajectories, window=2, step=2, boxfilter=False, verbose=False):

    to_split = {i:[] for i in range(len(trajectories))}
    for i, j in itertools.combinations(range(len(trajectories)), 2):
        traj1 = trajectories[i]
        traj2 = trajectories[j]     

        intersections = find_intersections_fast(traj1, traj2, window, step)
        if intersections and len(intersections):

            for x in intersections:
                to_split[i] += list(range(*x[0]))

            for x in intersections:
                to_split[j] += list(range(*x[1]))

    new_trajectories = []
    for i,idxs in to_split.items():
        new_trajectories += split_trajectory(trajectories[i], idxs, boxfilter=boxfilter)[0]
        
    return new_trajectories

def trim_endpoints_trajectories(trajectories, index_window=2, verbose=False):
    
    to_trim = {}
    for j, traj2 in enumerate(trajectories):
        for i, traj1 in enumerate(trajectories):
            if i!=j:

                for k in range(len(traj2)):
                    diff = traj1[-1].index-traj2[k].index
                    if diff>=0 and diff<index_window:
                        if j not in to_trim:
                            to_trim[j] = []
                        to_trim[j].append(k)
                    else:
                        break

    for j,to_delete in to_trim.items():
        _to_delete = list(set(to_delete))
        if verbose:
            print("[trim_endpoints_trajectories] trimming traj:{} {}".format(j,[trajectories[j][k] for k in _to_delete]))
        for k in reversed(_to_delete):
            trajectories[j].pop(k)
            
    return trajectories


def remove_parallel_trajectories(trajectories, th_similarity=0.7, index_window=5, dist=10, verbose=False):
    
    removed_trajectories = []
    remaining_trajectories = trajectories
    changed = True
    while changed:
        
        changed = False  
        for i, j in itertools.combinations(range(len(remaining_trajectories)), 2):
            traj1 = trajectories[i]
            traj2 = trajectories[j]

            s12_w, s21_w, idxs1, idxs2 = find_close_detections(traj1, traj2, index_window=index_window, dist=dist)

            if s12_w>th_similarity and s21_w>th_similarity:
                if (traj1[-1].index-traj1[0].index)>(traj2[-1].index-traj2[0].index):
                    remaining_trajectories.remove(traj2)
                    removed_trajectories.append(traj2)
                    changed = True
                else:
                    remaining_trajectories.remove(traj1)
                    removed_trajectories.append(traj1)
                    changed = True
                break   
                
    return remaining_trajectories, removed_trajectories