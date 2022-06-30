import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import json

class PlaneDetection:
    def __init__(self, calib_path, corners, box_z = 3.0):
        """
        PlaneDetection object constructor. Initializes data containers.
        
        """
        self.corners = corners
        self.box_z = box_z
        self.id_to_find  = 0
        self.marker_size  = 2 #cm
        self.homography = None
        
        self.box_vertices = {}
        self.tray_world_pts = {}
        self.tray_world_pts_detect = []
        self.tray_img_pts_detect = []
        self.tag_boxes = {
            '0': {'box':None,
                'pos':{
                '1':(1,5),
                '2':(2,6),
                '3':(3,7)}},
            '1': {'box':None,
                'pos':{
                '2':(1,5),
                '3':(2,6),
                '0':(3,7)}},
            '2': {'box':None,
                'pos':{
                '3':(1,5),
                '0':(2,6),
                '1':(3,7)}},
            '3': {'box':None,
                'pos':{
                '0':(1,5),
                '1':(2,6),
                '2':(3,7)}},
            }

        self.load_original_points()
        self.compute_tray_dims()
        self.define_boxes_for_tags()
        self.rotate_original_pts()

        self.camera_matrix = np.loadtxt(calib_path+'camera_matrix.txt', delimiter=',')
        self.camera_distortion = np.loadtxt(calib_path+'distortion.txt', delimiter=',')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

    def define_boxes_for_tags(self):
        self.define_template_tray_base()
        num_pts = len(self.corners)
        i = 0
        for iD, vector in self.tray_world_pts.items():
            if iD in self.corners.values():
                xyz_pt = np.append(np.array(vector), [0.0])/10
                xyz_pt_matrix = np.tile(xyz_pt, (num_pts, 1))
                ground_rect = self.template_tray_base - xyz_pt_matrix
                ground_rect_up_crop = ground_rect[:i,:]
                ground_rect_low_crop = ground_rect[i:,:]
                ground_rect = np.concatenate((ground_rect_low_crop, ground_rect_up_crop), axis=0)
                box_3d = np.concatenate((ground_rect, ground_rect), axis=0)
                box_3d[4:,2] = self.box_z
                self.tag_boxes[iD]['box'] = box_3d
                i+=1

        print(self.tag_boxes)
    
    def define_template_tray_base(self):
        tray_base = np.zeros((4,3))
        i = 0
        for (key, vector) in self.tray_world_pts.items():
            if key in self.corners.values():
                xyz_pt = np.append(np.array(vector), [0.0])/10
                tray_base[i] = xyz_pt
                i+=1
        self.template_tray_base = tray_base
        print('Template: \n',tray_base)
    
    def compute_tray_dims(self):
        
        tl = self.tray_world_pts[self.corners['tl']]
        tr = self.tray_world_pts[self.corners['tr']]
        br = self.tray_world_pts[self.corners['br']]
        bl = self.tray_world_pts[self.corners['bl']]
        
        self.tray_w = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))//10
        self.tray_h = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))//10

        print(self.tray_w, self.tray_h)
        

    def load_original_points(self):
        f = open('tray_points.json')
		# Dict of points in conveyor:
        self.tray_world_pts = json.load(f)

    def rotate_original_pts(self):
        Rot_x = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(math.radians(180)),-math.sin(math.radians(180))],
                    [0.0, math.sin(math.radians(180)), math.cos(math.radians(180))]])

        for key, vector in self.tray_world_pts.items():
            xyz_pt = np.append(np.array(vector), [0.0])
            self.tray_world_pts[key] = list(Rot_x @ xyz_pt)[:2]
        print(self.tray_world_pts)

    def compute_homog(self):
        self.homography = None
        self.tray_img_pts_detect = []
        self.tray_world_pts_detect = []
        for tag_id in self.box_vertices:
            if tag_id in self.tray_world_pts:
                self.tray_world_pts_detect.append(self.tray_world_pts[tag_id])
                self.tray_img_pts_detect.append(list(self.box_vertices[tag_id][0]))
        is_enough_points_detect = len(self.tray_img_pts_detect)>= 4
        if is_enough_points_detect:
            self.homography,status = cv2.findHomography(np.array(self.tray_img_pts_detect), 
												np.array(self.tray_world_pts_detect))
            return self.homography
        else:
            # print("[INFO]: Less than 4 corresponding points found")
            self.homography = None
            return self.homography
    
    def compute_perspective_trans(self, image, adaptive_aspect = False):
        corners = ['0','1','2','3']
        height, width, channels = image.shape
        
        if self.homography is not None and all(x in self.box_vertices for x in corners):
            tl = self.box_vertices['0'][0]
            tr = self.box_vertices['1'][0]
            br = self.box_vertices['2'][0]
            bl = self.box_vertices['3'][0]
            rect = (tl, tr, br, bl)
            # compute the width of the new image, which will be the
            # maximum distance between bottom-right and bottom-left
            # x-coordiates or the top-right and top-left x-coordinates
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            if adaptive_aspect:
                warped = cv2.warpPerspective(image, self.homography, (maxWidth, maxHeight))
            else:
                warped = cv2.warpPerspective(
                                        image, 
                                        self.homography, 
                                        (int(self.tray_w)*10, int(self.tray_h)*10))
            
            return warped
        else:
            return None

    def draw_tag_pose(self,image, rvec, tvec, z_rot=-1):
        world_points = np.array([
            4, 0, 0,
            0, 0, 0,
            0, 4, 0,
            0, 0, -4 * z_rot
        ]).reshape(-1, 1, 3) * 0.5 * self.marker_size

        img_points, _ = cv2.projectPoints(world_points, 
                                            rvec, tvec, 
                                            self.camera_matrix, 
                                            self.camera_distortion)
        img_points = np.round(img_points).astype(int)
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]

        cv2.line(image, img_points[0], img_points[1], (0,0,255), 2)
        cv2.line(image, img_points[1], img_points[2], (0,255,0), 2)
        cv2.line(image, img_points[1], img_points[3], (255,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'X', img_points[0], font, 0.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Y', img_points[2], font, 0.5, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(image, 'Z', img_points[3], font, 0.5, (255,0,0), 2, cv2.LINE_AA)
        cv2.putText(image, str((0,0)), (img_points[1][0]+10,img_points[1][1]-30), font, 0.5,
                                        (255, 255, 255), 1, cv2.LINE_AA)

    def define_world_pts(self,iD):
        world_points = self.tag_boxes[iD]['box']
        return world_points * 0.5 * self.marker_size

    def draw_box(self,image, iD , rvec, tvec):
        world_points = self.define_world_pts(str(iD), self.marker_size)
        img_points, _ = cv2.projectPoints(
                                world_points, 
                                rvec, tvec, 
                                self.camera_matrix, 
                                self.camera_distortion)

        img_points = np.round(img_points).astype(int)
        # img_points = [(x1,y1),(x2,y2),...] in pixels:
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)] 

        cv2.line(image, img_points[0], img_points[1], (255,0,0), 2)
        cv2.line(image, img_points[1], img_points[2], (255,0,0), 2)
        cv2.line(image, img_points[2], img_points[3], (255,0,0), 2)
        cv2.line(image, img_points[3], img_points[0], (255,0,0), 2)
        cv2.line(image, img_points[4], img_points[5], (255,0,0), 2)
        cv2.line(image, img_points[5], img_points[6], (255,0,0), 2)
        cv2.line(image, img_points[6], img_points[7], (255,0,0), 2)
        cv2.line(image, img_points[7], img_points[4], (255,0,0), 2)
        cv2.line(image, img_points[0], img_points[4], (255,0,0), 2)
        cv2.line(image, img_points[1], img_points[5], (255,0,0), 2)
        cv2.line(image, img_points[2], img_points[6], (255,0,0), 2)
        cv2.line(image, img_points[3], img_points[7], (255,0,0), 2)

    def compute_tag_z_vertices(self, rvec, tvec, z_rot=-1):
        world_points = np.array([
            0, 0, 0,
            0, 0, -3 * z_rot
        ]).reshape(-1, 1, 3) * 0.5 * self.marker_size

        img_points, _ = cv2.projectPoints(
                                world_points, 
                                rvec, tvec, 
                                self.camera_matrix, 
                                self.camera_distortion)

        img_points = np.round(img_points).astype(int)
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]

        return img_points[0],img_points[1]

    def rewrite_pts(self, pt_indices, box_vert_id ):
        pt_idx1 = pt_indices[0]
        pt_idx2 = pt_indices[1]
        self.points_update[pt_idx1] = self.box_vertices[box_vert_id][0]
        self.points_update[pt_idx2] = self.box_vertices[box_vert_id][1]

    def update_pts_w_id(self, iD):

        for box_vert_id in self.box_vertices:
            if box_vert_id in self.tag_boxes[iD]['pos']:
                self.rewrite_pts(self.tag_boxes[iD]['pos'][box_vert_id], box_vert_id)
            
    def update_img_pts(self, iD, rvec, tvec):

        self.points_update = [None,None,None,None,None,None,None,None]
        self.update_pts_w_id(iD)

        world_points = self.define_world_pts(iD)
        img_points, _ = cv2.projectPoints(
                                world_points, 
                                rvec, tvec, 
                                self.camera_matrix, 
                                self.camera_distortion)

        img_points = np.round(img_points).astype(int)
        # img_points = [(x1,y1),(x2,y2),...] in pixels:
        img_points = [self.points_update[i] if self.points_update[i] else tuple(pt) 
                        for i, pt in enumerate(img_points.reshape(-1, 2))] 
        return img_points

    def draw_box_update(self,image, iD, box_vertices, rvec, tvec):
        img_points = self.update_img_pts(str(iD), rvec, tvec)

        cv2.line(image, img_points[0], img_points[1], (255,0,0), 2)
        cv2.line(image, img_points[1], img_points[2], (255,0,0), 2)
        cv2.line(image, img_points[2], img_points[3], (255,0,0), 2)
        cv2.line(image, img_points[3], img_points[0], (255,0,0), 2)
        cv2.line(image, img_points[4], img_points[5], (255,0,0), 2)
        cv2.line(image, img_points[5], img_points[6], (255,0,0), 2)
        cv2.line(image, img_points[6], img_points[7], (255,0,0), 2)
        cv2.line(image, img_points[7], img_points[4], (255,0,0), 2)
        cv2.line(image, img_points[0], img_points[4], (255,0,0), 2)
        cv2.line(image, img_points[1], img_points[5], (255,0,0), 2)
        cv2.line(image, img_points[2], img_points[6], (255,0,0), 2)
        cv2.line(image, img_points[3], img_points[7], (255,0,0), 2)
        return img_points
    
    def detect_tags_3D(self, frame):
        self.box_vertices = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        corners, ids, rejected = cv2.aruco.detectMarkers(
                                            gray, 
                                            self.aruco_dict, 
                                            parameters = self.parameters,
                                            cameraMatrix = self.camera_matrix, 
                                            distCoeff = self.camera_distortion)

        if ids is not None and (self.id_to_find in ids):
            poses = cv2.aruco.estimatePoseSingleMarkers(
                                                corners, 
                                                self.marker_size, 
                                                self.camera_matrix, 
                                                self.camera_distortion)

            cv2.aruco.drawDetectedMarkers(frame, corners)
            grid_id = ids[0][0]
            self.rot_vecs, self.tran_vecs = poses[0], poses[1]
            self.box_vertices = {str(tag_id[0]):self.compute_tag_z_vertices( 
                                                            self.rot_vecs[i][0], 
                                                            self.tran_vecs[i][0]) 
                                                            for i, tag_id in enumerate(ids)}
            for i, tag_id in enumerate(ids):
                
                rvec , tvec = self.rot_vecs[i][0], self.tran_vecs[i][0]
                self.draw_tag_pose(frame, rvec, tvec)

                if tag_id == grid_id:

                    plane_img_pts = self.draw_box_update(
                                                    frame, 
                                                    str(grid_id), 
                                                    self.box_vertices, 
                                                    rvec, tvec)

calib_path = ""
corners = {
    'tl' :'0',
    'tr' :'1',
    'br' :'2',
    'bl' :'3'
    }

pd = PlaneDetection(calib_path, corners)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

    ret, frame = cap.read()
    raw_frame = frame.copy()
    pd.detect_tags_3D(frame)
    print(pd.box_vertices)
    homography = pd.compute_homog()
    frame_warp = pd.compute_perspective_trans(raw_frame)
            
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