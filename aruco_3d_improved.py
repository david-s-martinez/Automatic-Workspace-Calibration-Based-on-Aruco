import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import json
from plane_computation.plane_detection import PlaneDetection

cam_calib_paths = ('cam_matrix_pc_cam.txt','distortion_pc_cam.txt','plane_points.json')
corners = {
    'tl' :'0',
    'tr' :'1',
    'br' :'2',
    'bl' :'3'
    }
# tag_dict = cv2.aruco.DICT_APRILTAG_36h11
tag_dict = cv2.aruco.DICT_4X4_50
pd = PlaneDetection(cam_calib_paths, corners, marker_size=2, tag_scaling=0.5, box_z=2.6,tag_dict=tag_dict)
# With IP Cam:
# cap = cv2.VideoCapture('http://10.41.0.5:8080/?action=stream')

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:

    ret, frame = cap.read()
    raw_frame = frame.copy()
    pd.detect_tags_3D(frame)
    print(pd.box_verts_update)
    homography = pd.compute_homog(w_updated_pts=True)
    frame_warp = pd.compute_perspective_trans(raw_frame, w_updated_pts=True)
            
    cv2.imshow('frame', frame)
    if frame_warp is not None:
        w_height, w_width, w_channels = frame_warp.shape
        frame_warp = cv2.resize(frame_warp, (w_width*4, w_height*4))
        cv2.imshow('frame_warp', frame_warp)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break