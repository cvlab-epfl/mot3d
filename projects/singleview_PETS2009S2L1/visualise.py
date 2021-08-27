import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import os
import sys
import itertools
import imageio
import random
import numpy as np
import cv2
import sys
import time
import argparse
import shutil

import mot3d
from multiview_aircraft.utils import utils
from multiview_aircraft.tracking import feeders
from multiview_aircraft.tracking.scene import Scene

def draw_bbox(img, bbox, string, color, thickness_boxes):
    
    xmin, ymin, xmax, ymax = bbox
    
    a = 20
    b = 12*len(string)
    img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=color, thickness=thickness_boxes)
    
    return img

def draw_bbox_text(img, bbox, string, color, thickness_boxes):
    
    xmin, ymin, xmax, ymax = bbox
    
    a = 20
    b = 12*len(string)
    img = cv2.rectangle(img, (int(xmin), int(ymin)-a), (int(xmin)+b, int(ymin)), 
                        color=color, thickness=thickness_boxes)    
    img = cv2.rectangle(img, (int(xmin), int(ymin)-a), (int(xmin)+b, int(ymin)), 
                        color=color, thickness=-1)
    
    img = cv2.putText(img, str(string), (int(xmin+1), int(ymin)-6), 
                      fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                      fontScale=0.5, color=[255,255,255], thickness=2,
                      bottomLeftOrigin=False)
    
    return img

def visualisation(indexes, filenames, trajectories, colors=None, crop=(slice(None,None), slice(None,None)), 
                  calibration=None, trace_length=25, thickness=5, thickness_boxes=2,
                  output_path="./output/sequence1", output_video="sequence1.mp4", fps=25):
    
    from tqdm import tqdm
    import subprocess
    
    if colors is None:
        cs = [[255,0,0], [0,255,0], 
              [60,60,255], [255,255,0], 
              [0,255,255], [255,0,255],
              [225,225,225], [0,0,0],
              [128,128,128], [50,128,50]]+[np.random.randint(40,215,3).tolist() 
                                           for _ in range(len(trajectories)-10)]        
        colors = {class_:cs[i] for i,class_ in enumerate(trajectories.keys())}
              
    colors_trajs = {id(traj):np.random.randint(40,215,3).tolist()
                    for trajs in trajectories.values() for traj in trajs}        
    
    if crop is None:
        crop=(slice(None,None), slice(None,None))

    utils.mkdir(output_path) 
    
    trajs_indexes = {class_:[np.int32([d.index for d in traj]) for traj in trajs] 
                     for class_,trajs in trajectories.items()}
    trajs_ends = {class_:[(idxs.min(), idxs.max()) for idxs in indexes]
                  for class_,indexes in trajs_indexes.items()}    

    for i,filename in tqdm(zip(indexes,filenames)):

        basename = os.path.basename(filename)
        img = imageio.imread(filename)
        
        # write index
        img = cv2.putText(img, str(i), (int(10), int(25)), 
                          fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                          fontScale=0.7, color=[255,255,255], thickness=1,
                          bottomLeftOrigin=False) 
        
        for j,(class_,color) in enumerate(colors.items()):
            img = cv2.putText(img, str(class_), (int(10), int(50+25*j)), 
                              fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                              fontScale=0.7, color=color, thickness=1,
                              bottomLeftOrigin=False)     
              
        tracklets = []
        for class_, trajs in trajectories.items():
            
            for k,(traj, indexes, (idx_min, idx_max)) in enumerate(zip(trajectories[class_], 
                                                                       trajs_indexes[class_], 
                                                                       trajs_ends[class_])):
                
                if i>=idx_min and i<=idx_max:
                
                    indexes_diff = indexes-i
                    idxs_to_plot = np.where(np.logical_and(indexes_diff<0, indexes_diff>-trace_length))[0]
                    if len(idxs_to_plot):
                        
                        positions = np.float32([traj[i].position for i in idxs_to_plot])
                        
                        if  positions.shape[1]==3:
                            if calibration is None:
                                raise ValueError("Positions are three-dimensional but the calibration to project them is not provided!")
                            K = np.array(calibration['K'])
                            R = np.array(calibration['R'])
                            t = np.array(calibration['t'])
                            dist = np.array(calibration['dist'])
                            positions = cv2.projectPoints(positions, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1,2)
                        
                        tracklets.append({'positions':positions, 
                                          'color_class': colors[class_],
                                          'color_traj': colors_trajs[id(traj)],
                                          'bboxes':[getattr(traj[i], 'bbox', None) for i in idxs_to_plot],
                                          'id':k})

        for track in tracklets:
            
            for p0,p1 in zip(track['positions'][:-1], track['positions'][1:]):
                try:
                    img = cv2.line(img, tuple(p0), tuple(p1), color=track['color_traj'], 
                                   thickness=max(1, int(thickness*0.5)))
                except:
                    print(tuple(p0), tuple(p1), track['color_traj'], max(1, int(thickness*0.5)))
                    raise
                
            if track['bboxes'][-1] is not None:
                xmin, ymin, xmax, ymax = track['bboxes'][-1]
                #img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), 
                #                    color=track['color_bbox'], thickness=thickness_boxes)
                
                img = draw_bbox(img, (xmin, ymin, xmax, ymax), str(track['id']), track['color_traj'], thickness_boxes)
                
        for track in tracklets:
        
            if track['bboxes'][-1] is not None:
                xmin, ymin, xmax, ymax = track['bboxes'][-1]
                
                img = draw_bbox_text(img, (xmin, ymin, xmax, ymax), str(track['id']), track['color_traj'], thickness_boxes)                

        imageio.imwrite(os.path.join(output_path, basename), img[crop]) 
        
    if isinstance(output_video, str):
        ext = basename.split('.')[-1]
        cmd="ffmpeg -framerate {} -pattern_type glob -i '{}/*.{}' -vcodec libx264 -crf 19 -maxrate 3M -bufsize 6M -profile:v high -pix_fmt yuv420p -movflags faststart -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" {} -y".format(fps, output_path, ext, output_video)
        print(cmd)
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
        for line in out.stdout:
            print(line.strip())  

def main(config_file='',
         trace_length=25, 
         thickness=18, 
         thickness_boxes=2,
         fps=20,
         clear_output_folder=True,
         use_tracklets=False,
         use_smoothed=True,
         view=None):
    
    __c__ = utils.yaml_read(config_file)
    
    output_path = __c__['output_path']
    utils.mkdir(output_path)
    
    if 'views' in __c__:
        views = __c__['views']  
        if view is None:
            view = views[0]        
        feeder_name = __c__['feeder']['name']
        object_class = __c__['feeder'][feeder_name]['object_class']
        feeder = feeders.Feeder[feeder_name](view, **__c__['feeder'][feeder_name])
        iterator = iter(feeder)

        calibration = utils.json_read(__c__['calibration'])[view]     
    else:
        feeder_name = __c__['feeder']['name']
        object_class = __c__['feeder'][feeder_name]['object_class']
        feeder = feeders.Feeder[feeder_name](**__c__['feeder'][feeder_name])
        calibration = None
    
    scene = Scene()
    basename_config = os.path.splitext(os.path.basename(config_file))[0]
    if use_smoothed:
        output_file = os.path.join(output_path, "results_smooth_{}.pickle".format(basename_config))
    else:
        output_file = os.path.join(output_path, "results_{}.pickle".format(basename_config))
    scene.load(output_file)
    
    if use_tracklets:
        trajs = list(scene.tracklets.values())
    else:
        trajs = list(scene.completed.values())+list(scene.active.values())

    mot3d.plot_trajectories(trajs, display_time=True, display_time_every=10)
    if use_smoothed:
        plt.savefig(os.path.join(output_path, "plot_trajectories_smooth_{}.jpg".format(basename_config)), 
                    bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_path, "plot_trajectories_{}.jpg".format(basename_config)), 
                    bbox_inches='tight')                

    filenames = [feeder.get_filename_image(i) for i in range(len(feeder))]
    indexes = list(range(len(filenames)))

    trajectories = {object_class:trajs}
    
    if use_tracklets:
        output_frames=os.path.join(output_path, "{}_tracklets".format(basename_config))
        output_video=os.path.join(output_path, "{}_tracklets.mp4".format(basename_config))
    else:
        if use_smoothed:
            output_frames=os.path.join(output_path, "{}_smooth".format(basename_config))
            output_video=os.path.join(output_path, "{}_smooth.mp4".format(basename_config))
        else:
            output_frames=os.path.join(output_path, "{}".format(basename_config))
            output_video=os.path.join(output_path, "{}.mp4".format(basename_config))            
            
    utils.mkdir(output_frames)
    
    if clear_output_folder:
        print("Deleting content of folder {}".format(output_frames))
        shutil.rmtree(output_frames)

    visualisation(indexes, 
                  filenames, 
                  trajectories, 
                  crop=(slice(None,None), slice(None,None)), 
                  calibration=calibration,
                  trace_length=trace_length, 
                  thickness=thickness, 
                  thickness_boxes=thickness_boxes,
                  output_path=output_frames, 
                  output_video=output_video, 
                  fps=fps)
    
if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    

    parser = argparse.ArgumentParser()   
    parser.add_argument("--config_file", "-c", type=str, default="config.config")
    parser.add_argument("--trace_length", "-tl", type=int, default=25)
    parser.add_argument("--thickness", "-lw", type=int, default=18)
    parser.add_argument("--thickness_boxes", "-lwb", type=int, default=2)
    parser.add_argument("--fps", "-fps", type=float, default=25)
    parser.add_argument("--clear_output_folder", "-rm", type=str2bool, default='yes')
    parser.add_argument("--use_tracklets", action="store_true", required=False)
    parser.add_argument("--use_smoothed", action="store_true", required=False)
    parser.add_argument("--view", "-v", type=str, default=None, required=False)

    args = parser.parse_args()

    main(**vars(args))    
    
# python visualise.py -c config/config_singleview_PETS2009S2L1_View_001.yaml