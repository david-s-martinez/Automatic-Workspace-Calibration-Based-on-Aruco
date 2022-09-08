import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import json
from plane_computation.linked_list import Node, LinkedList

class PlaneDetection:
    def __init__(self, cam_calib_paths, corners, marker_size= 4,tag_scaling = 1, box_z = 3.0, tag_dict = cv2.aruco.DICT_4X4_50):
        """
        PlaneDetection object constructor. Initializes data containers.
        
        """
        self.cam_calib_paths = cam_calib_paths
        self.corners = corners
        self.box_z = box_z
        self.id_to_find  = 0
        self.marker_size  = marker_size #cm
        self.tag_scaling = tag_scaling
        self.homography = None
        self.Rot_x = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(math.radians(180)),-math.sin(math.radians(180))],
                    [0.0, math.sin(math.radians(180)), math.cos(math.radians(180))]])
        
        self.tag_boxes = {}
        self.box_vertices = {}
        self.plane_world_pts = {}
        self.plane_world_pts_detect = []
        self.plane_img_pts_detect = []
        
        self.__load_original_points()
        self.__compute_plane_dims()
        self.tag_order_linkd_list = LinkedList()
        self.__init_tag_boxes()
        self.__define_boxes_for_tags()
        self.__rotate_original_pts()

        self.camera_matrix = np.loadtxt(cam_calib_paths[0], delimiter=',')
        self.camera_distortion = np.loadtxt(cam_calib_paths[1], delimiter=',')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(tag_dict)
        self.parameters = cv2.aruco.DetectorParameters_create()
    
    def __load_original_points(self):
        f = open(self.cam_calib_paths[2])
		# Dict of points in conveyor:
        self.org_plane_world_pts = json.load(f)
        self.plane_world_pts = self.org_plane_world_pts.copy()
    
    def __compute_plane_dims(self):
        
        tl = self.plane_world_pts[self.corners['tl']]
        tr = self.plane_world_pts[self.corners['tr']]
        br = self.plane_world_pts[self.corners['br']]
        bl = self.plane_world_pts[self.corners['bl']]
        
        self.plane_w = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))/10
        self.plane_h = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))/10

        print(self.plane_w, self.plane_h)
    
    def __init_tag_boxes(self):
        for iD in self.plane_world_pts:
            self.tag_order_linkd_list.add_end(iD)
            self.tag_boxes[iD] = {'box':None,'pos':None}
        self.tag_order_linkd_list.make_circular()
        print(self.tag_order_linkd_list)

    def __compute_tag_relative_pos(self, iD, n):
        actual_node = self.tag_order_linkd_list.find_iD(iD)
        output = dict()
        num_pts = len(self.plane_world_pts)
        for i in range(n):
            output[actual_node.iD] = (i,i+num_pts)
            actual_node = actual_node.next
        return output
    
    def __define_template_plane_base(self):
        num_pts = len(self.plane_world_pts)
        plane_base = np.zeros((num_pts,3))

        for i, (key, vector) in enumerate(self.plane_world_pts.items()):
            # xyz_pt = np.append(np.array(vector), [0.0])/10
            xyz_pt = np.array(vector)/10
            plane_base[i] = xyz_pt

        self.template_plane_base = plane_base
        print('Template: \n',plane_base)

    def __define_boxes_for_tags(self):
        self.__define_template_plane_base()
        num_pts = len(self.plane_world_pts)
        
        for i,(iD, vector) in enumerate(self.plane_world_pts.items()):
            xyz_pt = np.array(vector)/10
            xyz_pt_matrix = np.tile(xyz_pt, (num_pts, 1))
            ground_rect = self.template_plane_base - xyz_pt_matrix
            ground_rect_up_crop = ground_rect[:i,:]
            ground_rect_low_crop = ground_rect[i:,:]
            ground_rect = np.concatenate((ground_rect_low_crop, ground_rect_up_crop), axis=0)
            box_3d = np.concatenate((ground_rect, ground_rect), axis=0)
            box_3d[num_pts:,2] = self.box_z
            positions = self.__compute_tag_relative_pos(iD, num_pts)
            self.tag_boxes[iD]['box'] = box_3d
            self.tag_boxes[iD]['pos'] = positions

        print(self.tag_boxes)
        
    def __rotate_original_pts(self):
        for key, vector in self.plane_world_pts.items():
            xyz_pt = np.array(vector)
            self.plane_world_pts[key] = list(self.Rot_x @ xyz_pt)
        print(self.plane_world_pts)

    def compute_homog(self, w_updated_pts = False, w_up_plane = False):
        self.homography = None
        self.plane_img_pts_detect = []
        self.plane_world_pts_detect = []
        box = self.box_verts_update if w_updated_pts else self.box_vertices
        for tag_id in box:
            if tag_id in self.plane_world_pts:
                self.plane_world_pts_detect.append(self.plane_world_pts[tag_id])
                verts_idx = 'top' if w_up_plane else 'base'
                self.plane_img_pts_detect.append(list(box[tag_id][verts_idx]))
        is_enough_points_detect = len(self.plane_img_pts_detect)>= 4
        if is_enough_points_detect:
            self.homography,status = cv2.findHomography(np.array(self.plane_img_pts_detect), 
												np.array(self.plane_world_pts_detect))
            return self.homography
        else:
            # print("[INFO]: Less than 4 corresponding points found")
            self.homography = None
            return self.homography
    
    def compute_perspective_trans(self, image, w_updated_pts = False, w_up_plane = False, adaptive_aspect = False):
        height, width, channels = image.shape
        box = self.box_verts_update if w_updated_pts else self.box_vertices
        if self.homography is not None and all(x in box for x in self.corners.values()):
            verts_idx = 'top' if w_up_plane else 'base'
            tl = box[self.corners['tl']][verts_idx]
            tr = box[self.corners['tr']][verts_idx]
            br = box[self.corners['br']][verts_idx]
            bl = box[self.corners['bl']][verts_idx]
            # compute the width of the new image, which will be the
            # maximum distance between bottom-right and bottom-left
            # x-coordiates or the top-right and top-left x-coordinates
            try:
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
                                            (int(self.plane_w)*10, int(self.plane_h)*10))
            
                return warped
            except:
                return None
        else:
            return None

    def draw_tag_pose(self,image, rvec, tvec, tag_id, z_rot=-1):
        world_points = np.array([
            0, 0, 0,
            4, 0, 0,
            0, 4, 0,
            0, 0, -4 * z_rot,
            1,1,0
        ]).reshape(-1, 1, 3) * self.tag_scaling * self.marker_size

        img_points, _ = cv2.projectPoints(world_points, 
                                            rvec, tvec, 
                                            self.camera_matrix, 
                                            self.camera_distortion)
        img_points = np.round(img_points).astype(int)
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]

        cv2.line(image, img_points[0], img_points[1], (0,0,255), 2)
        cv2.line(image, img_points[0], img_points[2], (0,255,0), 2)
        cv2.line(image, img_points[0], img_points[3], (255,69,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'X', img_points[1], font, 0.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Y', img_points[2], font, 0.5, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(image, 'Z', img_points[3], font, 0.5, (255,69,0), 2, cv2.LINE_AA)
        cv2.putText(image, str((tag_id)), (img_points[4][0],img_points[4][1]), font, 0.5,
                                        (255,255,0), 2, cv2.LINE_AA)

    def __define_world_pts(self,iD):
        world_points = self.tag_boxes[iD]['box']
        return world_points * self.tag_scaling * self.marker_size

    def draw_box(self,image, iD , rvec, tvec):
        world_points = self.__define_world_pts(str(iD), self.marker_size)
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
        
    def __refine_tag_pose(self, iD, ids, corners, rvec, tvec, w_updated_pts = False, w_up_plane = False):
        
        plane_img_pts_detect = []
        plane_world_pts_detect = []
        box = self.tag_boxes[str(iD)]['box']
        pos = self.tag_boxes[str(iD)]['pos']
        # If nothing was detected, return
        tag_corner_list = corners
        tag_id_list = ids.flatten()
        
        for (tag_corners, tag_id) in zip(tag_corner_list, tag_id_list):
            tag_id = str(tag_id)
            # Get (x, y) corners of the tag
            corners = tag_corners.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners

            top_left = (int(top_left[0]), int(top_left[1]))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

            # Compute centroid
            cX = int((top_left[0] + bottom_right[0]) / 2.0)
            cY = int((top_left[1] + bottom_right[1]) / 2.0)

            # Store detected points for homography computation
            centroid = [cX, cY]

            if tag_id in self.plane_world_pts:
                plane_world_pts_detect.append(list(box[pos[tag_id][0]]))
                plane_img_pts_detect.append(centroid)
        
        is_enough_points_detect = len(plane_img_pts_detect)>= 4
        try:
            new_rvec, new_tvec = cv2.solvePnPRefineLM(np.array(plane_world_pts_detect,dtype=np.float32), 
                                                np.array(plane_img_pts_detect,dtype=np.float32), 
                                                self.camera_matrix, self.camera_distortion,
                                                rvec, tvec)
            return new_rvec, new_tvec
        except Exception as e:
            return rvec, tvec 

    def __compute_tag_axis(self, idx, ids, corners, rvecs, tvecs, z_rot=-1, correct_Z_flip = False):
        rvec = rvecs[idx][0]
        tvec = tvecs[idx][0]
        iD = ids[idx][0]
        
        rvec,tvec = self.__refine_tag_pose(iD, ids, corners, rvec, tvec)
        
        world_points = np.array([
            0, 0, 0,
            0, 0, -3 * z_rot
        ]).reshape(-1, 1, 3) * self.tag_scaling * self.marker_size

        img_points, _ = cv2.projectPoints(
                                world_points, 
                                rvec, tvec, 
                                self.camera_matrix, 
                                self.camera_distortion)

        img_points = np.round(img_points).astype(int)
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]
        return {'base':img_points[0],'top':img_points[1], 'rvec':rvec, 'tvec':tvec}

    def __rewrite_pts(self, pt_indices, box_vert_id ):
        pt_idx1 = pt_indices[0]
        pt_idx2 = pt_indices[1]
        self.pts_update[pt_idx1] = self.box_vertices[box_vert_id]['base']
        self.pts_update[pt_idx2] = self.box_vertices[box_vert_id]['top']

    def __update_pts_w_id(self, iD):

        for box_vert_id in self.box_vertices:
            if box_vert_id in self.tag_boxes[iD]['pos']:
                self.__rewrite_pts(self.tag_boxes[iD]['pos'][box_vert_id], box_vert_id)
            
    def __update_img_pts(self, iD, rvec, tvec):
        num_pts = len(self.plane_world_pts)
        self.pts_update = [None] * num_pts * 2
        self.__update_pts_w_id(iD)

        world_points = self.__define_world_pts(iD)
        img_points, _ = cv2.projectPoints(
                                world_points, 
                                rvec, tvec, 
                                self.camera_matrix, 
                                self.camera_distortion)
        #TODO get translation vectors of all tags with world pts and given rvec 

        img_points = np.round(img_points).astype(int).reshape(-1, 2)
        # box_verts_update = {'id1':(x1,y1),'id2':(x2,y2),...} in pixels:
        box_verts_update = {
            key:{'base':self.pts_update[i], 'top':self.pts_update[i+num_pts]} if self.pts_update[i] 
            else {'base':tuple(img_points[i]),'top':tuple(img_points[i+num_pts])} 
            for i, key in enumerate(self.tag_boxes[iD]['pos'])
            }
        return box_verts_update

    def __compute_box_update(self,image, iD, rvec, tvec):
        box_update = self.__update_img_pts(str(iD), rvec, tvec)

        cv2.line(image, 
            box_update[self.corners['tl']]['base'], 
            box_update[self.corners['tr']]['base'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['tr']]['base'], 
            box_update[self.corners['br']]['base'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['br']]['base'], 
            box_update[self.corners['bl']]['base'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['bl']]['base'], 
            box_update[self.corners['tl']]['base'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['tl']]['base'], 
            box_update[self.corners['tl']]['top'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['tr']]['base'], 
            box_update[self.corners['tr']]['top'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['br']]['base'], 
            box_update[self.corners['br']]['top'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['bl']]['base'], 
            box_update[self.corners['bl']]['top'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['tl']]['top'], 
            box_update[self.corners['tr']]['top'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['tr']]['top'], 
            box_update[self.corners['br']]['top'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['br']]['top'], 
            box_update[self.corners['bl']]['top'], (0,165,255), 2)
        cv2.line(image, 
            box_update[self.corners['bl']]['top'], 
            box_update[self.corners['tl']]['top'], (0,165,255), 2)
        return box_update

    def __compute_plane_origin(self, frame, rvec,tvec, 
                        z_rot=1, cam_pose = False):
        world_points = np.array([
            6.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 6.0, 0.0,
            0.0, 0.0, 6.0 * z_rot
        ]).reshape(-1, 1, 3)
        x_plane, y_plane, z_plane = tvec[0], tvec[1], tvec[2]
        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, self.camera_matrix, self.camera_distortion)
        img_points = np.round(img_points).astype(int)
        img_points = [tuple(pt) for pt in img_points.reshape(-1, 2)]

        cv2.line(frame, img_points[0], img_points[1], (0,0,255), 2)
        cv2.line(frame, img_points[1], img_points[2], (0,255,0), 2)
        cv2.line(frame, img_points[1], img_points[3], (255,0,0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'X', img_points[0], font, 0.5, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Y', img_points[2], font, 0.5, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Z', img_points[3], font, 0.5, (255,0,0), 2, cv2.LINE_AA)
        
    def compute_plane_pose(self, frame, w_updated_pts = False, w_up_plane = False):
        
        plane_img_pts_detect = []
        plane_world_pts_detect = []
        box = self.box_verts_update if w_updated_pts else self.box_vertices
        for tag_id in box:
            if tag_id in self.org_plane_world_pts:
                plane_world_pts_detect.append([self.org_plane_world_pts[tag_id][0]/10,self.org_plane_world_pts[tag_id][1]/10,0.0])
                verts_idx = 'top' if w_up_plane else 'base'
                plane_img_pts_detect.append(list(box[tag_id][verts_idx]))
                if True:
                    cv2.putText(frame, str(self.org_plane_world_pts[tag_id]), box[tag_id][verts_idx],
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        is_enough_points_detect = len(plane_img_pts_detect)>= 4
        if is_enough_points_detect:
            _,rvec, tvec = cv2.solvePnP(np.array(plane_world_pts_detect,dtype=np.float32), 
                                        np.array(plane_img_pts_detect,dtype=np.float32), 
                                        self.camera_matrix, self.camera_distortion)
            self.__compute_plane_origin(frame, rvec, tvec)
            return rvec, tvec
        else:
            return None, None 

    def detect_tags_3D(self, frame):
        self.box_vertices = {}
        self.box_verts_update = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        corners, ids, rejected = cv2.aruco.detectMarkers(
                                            gray, 
                                            self.aruco_dict, 
                                            parameters = self.parameters,
                                            cameraMatrix = self.camera_matrix, 
                                            distCoeff = self.camera_distortion)

        # if ids is not None and (self.id_to_find in ids):
        if ids is not None :
            poses = cv2.aruco.estimatePoseSingleMarkers(
                                                corners, 
                                                self.marker_size, 
                                                self.camera_matrix, 
                                                self.camera_distortion)

            min_id = min(ids[0])
            self.rot_vecs, self.tran_vecs = poses[0], poses[1]
            self.box_vertices = {str(tag_id[0]):self.__compute_tag_axis(
                                                            i,
                                                            ids,
                                                            corners,
                                                            self.rot_vecs, 
                                                            self.tran_vecs) 
                                                            for i, tag_id in enumerate(ids)}
            cv2.aruco.drawDetectedMarkers(frame, corners)
            for i, tag_id in enumerate(ids):
                
                rvec , tvec = self.rot_vecs[i][0], self.tran_vecs[i][0]
                self.draw_tag_pose(frame, rvec, tvec, tag_id)

                if tag_id == min_id and str(min_id) in self.plane_world_pts.keys():

                    self.box_verts_update = self.__compute_box_update(
                                                    frame, 
                                                    str(min_id),
                                                    rvec, tvec)