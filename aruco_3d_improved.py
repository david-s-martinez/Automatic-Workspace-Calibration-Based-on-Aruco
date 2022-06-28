import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import json

class PlaneDetection:
    def __init__(self, calib_path, grid_w, grid_h,):
        """
        PlaneDetection object constructor. Initializes data containers.
        
        """
        self.id_to_find  = 4
        self.marker_size  = 2 #cm
        self.camera_matrix = np.loadtxt(calib_path+'camera_matrix.txt', delimiter=',')
        self.camera_distortion = np.loadtxt(calib_path+'distortion.txt', delimiter=',')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()
        
        self.cube_vertices = {}
        self.world_points = {}
        self.homography = None
        self.world_points_detect = []
        self.image_points_detect = []

        self.tag_cubes = {
            '0': {'cube':[
                0, 0, 0,
                grid_w, 0, 0,
                grid_w, -grid_h, 0,
                0, -grid_h, 0,
                0, 0, 3,
                grid_w, 0, 3,
                grid_w, -grid_h, 3,
                0, -grid_h, 3],
                'pos':{
                '1':(1,5),
                '2':(2,6),
                '3':(3,7)}},

            '1': {'cube':[
                0, 0, 0,
                0, -grid_h, 0,
                -grid_w, -grid_h, 0,
                -grid_w, 0, 0,
                0, 0, 3,
                0, -grid_h, 3,
                -grid_w, -grid_h, 3,
                -grid_w, 0, 3],
                'pos':{
                '2':(1,5),
                '3':(2,6),
                '0':(3,7)}},

            '2': {'cube':[
                0, 0, 0,
                -grid_w, 0, 0,
                -grid_w, grid_h, 0,
                0, grid_h, 0,
                0, 0, 3,
                -grid_w, 0, 3,
                -grid_w, grid_h, 3,
                0, grid_h, 3],
                'pos':{
                '3':(1,5),
                '0':(2,6),
                '1':(3,7)}},

            '3': {'cube':[
                0, 0, 0,
                0, grid_h, 0,
                grid_w, grid_h, 0,
                grid_w, 0, 0,
                0, 0, 3,
                0, grid_h, 3,
                grid_w, grid_h, 3,
                grid_w, 0, 3],
                'pos':{
                '0':(1,5),
                '1':(2,6),
                '2':(3,7)}},

            '4': {'cube':[
                -12.75, 7, 0,
                12.75, 7, 0,
                12.75, -7, 0,
                -12.75, -7, 0,
                -12.75, 7, 3,
                12.75, 7, 3,
                12.75, -7, 3,
                -12.75, -7, 3],
                'pos':{
                '0':(0,4),
                '1':(1,5),
                '2':(2,6),
                '3':(3,7)}}
            }
        self.load_original_points()

    def load_original_points(self):
        f = open('tray_points.json')
		# Dict of points in conveyor:
        self.world_points = json.load(f)

    def compute_homog(self):
        self.homography = None
        self.image_points_detect = []
        self.world_points_detect = []
        for tag_id in self.cube_vertices:
            if tag_id in self.world_points:
                self.world_points_detect.append(self.world_points[tag_id])
                self.image_points_detect.append(list(self.cube_vertices[tag_id][0]))
        is_enough_points_detect = len(self.image_points_detect)>= 4
        if is_enough_points_detect:
            self.homography,status = cv2.findHomography(np.array(self.image_points_detect), 
												np.array(self.world_points_detect))
            return self.homography
        else:
            # print("[INFO]: Less than 4 corresponding points found")
            self.homography = None
            return self.homography
    
    def compute_perspective_trans(self, image, adaptive_aspect = False):
        corners = ['0','1','2','3']
        height, width, channels = image.shape
        
        if self.homography is not None and all(x in self.cube_vertices for x in corners):
            tl = self.cube_vertices['0'][0]
            tr = self.cube_vertices['1'][0]
            br = self.cube_vertices['2'][0]
            bl = self.cube_vertices['3'][0]
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
                warped = cv2.warpPerspective(image, self.homography, (width//4, height//4))
            
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
        world_points = np.array(self.tag_cubes[iD]['cube']).reshape(-1, 1, 3)
        return world_points * 0.5 * self.marker_size

    def draw_cube(self,image, iD , rvec, tvec):
        world_points = self.define_world_pts(str(iD), self.marker_size)
        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, self.camera_matrix, self.camera_distortion)
        img_points = np.round(img_points).astype(int)
        
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)] # -> [(x1,y1),(x2,y2),...] in pixels

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

        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, self.camera_matrix, self.camera_distortion)
        img_points = np.round(img_points).astype(int)
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]

        return img_points[0],img_points[1]

    def rewrite_pts(self, pt_indices, cube_vert_id ):
        pt_idx1 = pt_indices[0]
        pt_idx2 = pt_indices[1]
        self.points_update[pt_idx1] = self.cube_vertices[cube_vert_id][0]
        self.points_update[pt_idx2] = self.cube_vertices[cube_vert_id][1]

    def update_pts_w_id(self, iD):

        for cube_vert_id in self.cube_vertices:
            if cube_vert_id in self.tag_cubes[iD]['pos']:
                self.rewrite_pts(self.tag_cubes[iD]['pos'][cube_vert_id], cube_vert_id)
            
    def update_img_pts_dict(self, iD, rvec, tvec):

        self.points_update = [None,None,None,None,None,None,None,None]
        self.update_pts_w_id(iD)

        world_points = self.define_world_pts(iD)
        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, self.camera_matrix, self.camera_distortion)
        img_points = np.round(img_points).astype(int)
        img_points = [ self.points_update[i] if self.points_update[i] else tuple(pt) for i, pt in enumerate(img_points.reshape(-1, 2))] # -> [(x1,y1),(x2,y2),...] in pixels
        
        return img_points

    def update_img_pts(self,iD, cube_vertices, rvec, tvec):
        
        points_update = [None,None,None,None,None,None,None,None]
        vert_2_update = list(cube_vertices.keys())

        if iD == '0':
            if '1' in vert_2_update:
                points_update[1] = cube_vertices['1'][0]
                points_update[5] = cube_vertices['1'][1]
            if '2' in vert_2_update:
                points_update[2] = cube_vertices['2'][0]
                points_update[6] = cube_vertices['2'][1]
            if '3' in vert_2_update:
                points_update[3] = cube_vertices['3'][0]
                points_update[7] = cube_vertices['3'][1]
            
        if iD == '1':
            if '2' in vert_2_update:
                points_update[1] = cube_vertices['2'][0]
                points_update[5] = cube_vertices['2'][1]
            if '3' in vert_2_update:
                points_update[2] = cube_vertices['3'][0]
                points_update[6] = cube_vertices['3'][1]
            if '0' in vert_2_update:
                points_update[3] = cube_vertices['0'][0]
                points_update[7] = cube_vertices['0'][1]
            
        if iD == '2':
            if '3' in vert_2_update:
                points_update[1] = cube_vertices['3'][0]
                points_update[5] = cube_vertices['3'][1]
            if '0' in vert_2_update:
                points_update[2] = cube_vertices['0'][0]
                points_update[6] = cube_vertices['0'][1]
            if '1' in vert_2_update:
                points_update[3] = cube_vertices['1'][0]
                points_update[7] = cube_vertices['1'][1]
            
        if iD == '3':
            if '0' in vert_2_update:
                points_update[1] = cube_vertices['0'][0]
                points_update[5] = cube_vertices['0'][1]
            if '1' in vert_2_update:
                points_update[2] = cube_vertices['1'][0]
                points_update[6] = cube_vertices['1'][1]
            if '2' in vert_2_update:
                points_update[3] = cube_vertices['2'][0]
                points_update[7] = cube_vertices['2'][1]
            
        if iD == '4':
            if '0' in vert_2_update:
                points_update[0] = cube_vertices['0'][0]
                points_update[4] = cube_vertices['0'][1]
            if '1' in vert_2_update:
                points_update[1] = cube_vertices['1'][0]
                points_update[5] = cube_vertices['1'][1]
            if '2' in vert_2_update:
                points_update[2] = cube_vertices['2'][0]
                points_update[6] = cube_vertices['2'][1]
            if '3' in vert_2_update:
                points_update[3] = cube_vertices['3'][0]
                points_update[7] = cube_vertices['3'][1]

        world_points = self.define_world_pts(iD)
        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, self.camera_matrix, self.camera_distortion)
        img_points = np.round(img_points).astype(int)
        img_points = [ points_update[i] if points_update[i] else tuple(pt) for i, pt in enumerate(img_points.reshape(-1, 2))] # -> [(x1,y1),(x2,y2),...] in pixels
        
        return img_points

    def draw_cube_update(self,image, iD, cube_vertices, rvec, tvec):
        # img_points = self.update_img_pts(str(iD), cube_vertices, rvec, tvec)
        img_points = self.update_img_pts_dict(str(iD), rvec, tvec)

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
        self.cube_vertices = {}
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
            self.cube_vertices = {str(tag_id[0]):self.compute_tag_z_vertices( 
                                                            self.rot_vecs[i][0], 
                                                            self.tran_vecs[i][0]) 
                                                            for i, tag_id in enumerate(ids)}
            for i, tag_id in enumerate(ids):
                
                rvec , tvec = self.rot_vecs[i][0], self.tran_vecs[i][0]
                self.draw_tag_pose(frame, rvec, tvec)

                if tag_id == grid_id:

                    plane_img_pts = self.draw_cube_update(frame, str(grid_id), self.cube_vertices, rvec, tvec)

	
calib_path = ""
# grid_w = 23.5
# grid_h = 13.0
grid_w = 25.5
grid_h = 14.0
pd = PlaneDetection(calib_path, grid_w, grid_h)

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

    ret, frame = cap.read()
    frame_warp = frame.copy()
    pd.detect_tags_3D(frame)
    print(pd.cube_vertices)
    homography = pd.compute_homog()
    frame_persp_trans = pd.compute_perspective_trans(frame_warp)
            
    cv2.imshow('frame', frame)
    if frame_persp_trans is not None:
        cv2.imshow('frame_persp_trans', frame_persp_trans)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break