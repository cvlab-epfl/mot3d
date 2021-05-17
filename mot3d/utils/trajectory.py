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