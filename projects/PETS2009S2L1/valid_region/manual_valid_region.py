#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2021
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import os
import itertools
import imageio

from mot3d.utils import utils
from mot3d.utils.singleview_geometry import project_points, project_points_homography, warpPerspectiveFrontal

output_path = './manual'

views = ['View_00'+str(i) for i in [1,3,4,5,6,7,8]]
filename_image = "/cvlabsrc1/cvlab/datasets_people_tracking/Crowd_PETS09/S2/L1/Time_12-34/{}/frame_0000.jpg"
filename_calibration = "../calibration.json"
filename_calibration = "/cvlabsrc1/cvlab/datasets_people_tracking/Crowd_PETS09/calibrations/calib/output/global_registration/global_poses.json"

# rectangle defining the valid region in world coordinate
xmin, xmax = -14000, -2000
ymin, ymax = -14000, -3000
zmin, zmax = -100, 1800

if __name__=='__main__':
    
    utils.mkdir(output_path)   
    
    calibration = utils.json_read(filename_calibration)
    
    N = 100
    grid = np.meshgrid(np.linspace(xmin, xmax, N),
                       np.linspace(ymin, ymax, N),
                       np.linspace(zmin, zmax, N))
    grid = np.vstack([x.ravel() for x in grid]).T.astype(np.float32)
    print("xmin:{} ymin:{} xmax:{} ymax:{}".format(xmin, ymin, xmax, ymax))
    print("grid:", grid.shape)
        
    valid_regions = {'ground':[[xmin,ymin],
                               [xmin,ymax],
                               [xmax,ymax],
                               [xmax,ymin]]}
    for view in views:
        img = imageio.imread(filename_image.format(view))
        
        K = np.array(calibration[view]['K'])
        R = np.array(calibration[view]['R'])
        t = np.array(calibration[view]['t'])
        dist = None#np.array(calibration[view]['dist'])

        proj, mask = project_points(grid, K, R, t, dist, image_shape=img.shape)
        proj = cv2.convexHull(proj[mask], returnPoints=True).squeeze()
                                    
        valid_regions[view] = proj.tolist()

        # visualisation
        mask_polygon = np.zeros(img.shape, np.uint8)
        mask_polygon = cv2.fillPoly(mask_polygon, [np.int32(proj)], [255,0,0])

        alpha = 0.7
        image_sup = cv2.addWeighted(img, alpha, mask_polygon, 1-alpha, 0)
        imageio.imwrite(os.path.join(output_path, "img_sup_{}.jpg".format(view)), image_sup)     
            
    utils.json_write(os.path.join(output_path, "valid_regions.json"), valid_regions) 