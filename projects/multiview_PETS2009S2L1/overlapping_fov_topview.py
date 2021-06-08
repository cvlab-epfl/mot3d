import matplotlib
matplotlib.use('Agg')
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
import itertools
import imageio

from mot3d.utils import utils
from mot3d.utils.singleview_geometry import project_points_homography, warpPerspectiveFrontal

output_path = './overlapping_fov'

views = ['View_00'+str(i) for i in [1,3,4,5,6,7,8]]
filename_image = "/cvlabsrc1/cvlab/datasets_people_tracking/Crowd_PETS09/S2/L1/Time_12-34/{}/frame_0000.jpg"
filename_calibration = "./calibration.json"

# region convered by at least this number of cameras
N_intersection = 3

margin_image = 50

# rectangle defining the ground scene in world coordinate
xmin, xmax = -16000, 16000
ymin, ymax = -16000, 30000
width, height = xmax-xmin, ymax-ymin

# rectangle defining the template/topview image in pixels
xmin_t, xmax_t = (xmin+16000)/20, (xmax+16000)/20
ymin_t, ymax_t = (ymin+16000)/20, (ymax+16000)/20
width_t, height_t = xmax_t-xmin_t, ymax_t-ymin_t

if __name__=='__main__':
    
    utils.mkdir(output_path)
    
    src = np.float32([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]
    ])
    dst = np.float32([
        [xmin_t, ymin_t],
        [xmin_t, ymax_t],
        [xmax_t, ymax_t],
        [xmax_t, ymin_t]
    ])
    size_topview = [height_t, width_t]

    M = cv2.findHomography(src, dst)[0]    
    
    calibration = utils.json_read(filename_calibration)
    
    regions = []
    for view in views:
        img = imageio.imread(filename_image.format(view))
        
        K = np.array(calibration[view]['K'])
        R = np.array(calibration[view]['R'])
        t = np.array(calibration[view]['t'])
        
        Hr = np.dot(K, np.hstack([R[:, :2], t[:, None]]))
        
        H = np.dot(Hr, np.linalg.inv(M))
        
        mask_image = np.ones(img.shape[:2])
        mask_image[:margin_image, :] = 0
        mask_image[-margin_image:, :] = 0
        mask_image[:margin_image,:] = 0
        mask_image[:,-margin_image:] = 0
        
        warp = warpPerspectiveFrontal(mask_image, np.linalg.inv(H), size_topview[::-1])    
        regions.append(warp)
        
        warp_image = warpPerspectiveFrontal(img, np.linalg.inv(H), size_topview[::-1])
        imageio.imwrite(os.path.join(output_path, "topview_{}.jpg".format(view)), warp_image)
        
    region = np.sum(regions, axis=0)>=N_intersection
    
    contours,_ = cv2.findContours(np.uint8(region), 1, 2)
    contour = contours[0].squeeze()   

    contour = project_points_homography(np.linalg.inv(M), contour)

    utils.json_write(os.path.join(output_path, "overlapping_fov_n{}.json".format(N_intersection)), 
                     {'contour':contour.tolist(), 'M':M.tolist()})
    
    # Visualisation
    for view in views:
        warp_image = imageio.imread(os.path.join(output_path, "topview_{}.jpg".format(view)))

        proj = project_points_homography(M, contour)

        mask_polygon = np.zeros(warp_image.shape, np.uint8)
        mask_polygon = cv2.fillPoly(mask_polygon, [np.int32(proj)], [255,0,0])

        alpha = 0.7
        warp_image = cv2.addWeighted(warp_image, alpha, mask_polygon, 1-alpha, 0)

        imageio.imwrite(os.path.join(output_path, "topview_sup_{}.jpg".format(view)), warp_image)