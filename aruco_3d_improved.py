import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import json
from plane_computation.plane_detection import PlaneDetection

# cam_source = 0
cam_source = 'delta_robot.mp4'
# cam_source = 'http://10.41.0.4:8080/?action=stream'
url_detections = 'http://10.41.0.4:5000/detections'
CAM_CONFIG_PATH = './detection_config/'
MODEL_PATH = './yoloV4_config/'
TAG_TYPE = 'april'
# TAG_TYPE = 'og_aruco'
CAM_TYPE = 'rpi'

path_dict = {
'cam_matrix':{'rpi':CAM_CONFIG_PATH+'camera_matrix_rpi.txt',
                'pc':CAM_CONFIG_PATH+'camera_matrix_pc_cam.txt' },

'distortion':{'rpi':CAM_CONFIG_PATH+'distortion_rpi.txt',
                'pc':CAM_CONFIG_PATH+'distortion_pc_cam.txt' },

'plane_pts':{'april':CAM_CONFIG_PATH+'plane_points_new_tray.json',
                'og_aruco':CAM_CONFIG_PATH+'new_aruco_tray.json',
                'aruco':CAM_CONFIG_PATH+'plane_points_old_tray.json'},

'model' : {'model_config':MODEL_PATH+'config.cfg',
                'weights':MODEL_PATH+'yolo.weights'},
    }

plane_config = {
'tag_dicts' : {'aruco':cv2.aruco.DICT_4X4_50,
            'og_aruco':cv2.aruco.DICT_ARUCO_ORIGINAL,
            'april':cv2.aruco.DICT_APRILTAG_36h11},

'plane_corners' : {'og_aruco': {'tl' :'0','tr' :'9','br' :'21','bl' :'12'}, 
                    'aruco': {'tl' :'0','tr' :'1','br' :'2','bl' :'3'},
                    'april':{'tl' :'30','tr' :'101','br' :'5','bl' :'6'}},
# 'plane_corners' : {'aruco': {'tl' :'0','tr' :'1','br' :'2','bl' :'3'}, 
#                     'april':{'tl' :'30','tr' :'101','br' :'5','bl' :'6'}},
    }

config = {
'neural_net': path_dict['model'],

'vision': (path_dict['cam_matrix'][CAM_TYPE],
            path_dict['distortion'][CAM_TYPE],
            path_dict['plane_pts'][TAG_TYPE]),

'plane': {'tag_size' : 2.86,
        'tag_scaling' : 0.36,
        'z_tansl' : 2.55,
        'tag_dict': plane_config['tag_dicts'][TAG_TYPE],
        'corners': plane_config['plane_corners'][TAG_TYPE]}
    }
pd = PlaneDetection(config['vision'], 
                        config['plane']['corners'], 
                        marker_size = config['plane']['tag_size'], 
                        tag_scaling = config['plane']['tag_scaling'], 
                        box_z = config['plane']['z_tansl'],
                        tag_dict = config['plane']['tag_dict'])

cap = cv2.VideoCapture(cam_source)

while True:

    ret, frame = cap.read()
    raw_frame = frame.copy()
    
    pd.detect_tags_3D(frame)
    homography = pd.compute_homog(w_updated_pts=True, w_up_plane=True)
    frame_warp = pd.compute_perspective_trans(raw_frame, w_updated_pts=True, w_up_plane=True)
    plane_rot, plane_trans = pd.compute_plane_pose(raw_frame)

    # print(pd.)
    f_height, f_width, f_channels = frame.shape
    frame = cv2.resize(frame, (f_width*2, f_height*2))
    raw_frame = cv2.resize(raw_frame, (f_width*2, f_height*2))
    cv2.imshow('frame', frame)
    cv2.imshow('raw_frame', raw_frame)
    
    if frame_warp is not None:
        w_height, w_width, w_channels = frame_warp.shape
        frame_warp = cv2.resize(frame_warp, (w_width*4, w_height*4))
        cv2.imshow('frame_warp', frame_warp)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break