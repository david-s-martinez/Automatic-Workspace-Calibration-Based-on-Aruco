import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math
import json

class Node:
    def __init__(self, iD = None, next_node = None):
        self.iD = iD
        self.next = next_node

class LinkedList:
    def __init__(self,head = None):
        self.head = head
        self.circular_lenght = None

    def is_head_undefined(self):
        return self.head is None

    def __str__(self):
        if self.is_head_undefined():
            return "Empty linked list"
        actual_node = self.head
        list_str = ''
        if self.circular_lenght is None:
            while actual_node:
                list_str += str(actual_node.iD)+'-> '
                actual_node = actual_node.next
        else:
            for i in range((self.circular_lenght+1)*2):
                list_str += str(actual_node.iD)+'-> '
                actual_node = actual_node.next
        return list_str

    def find_iD(self, iD):
        actual_node = self.head
        while actual_node:
            if actual_node.iD == iD:
                print('Found!')
                return actual_node
            actual_node = actual_node.next

    def add_end(self,iD):
        if self.is_head_undefined():
            self.head = Node(iD = iD)
        else:    
            actual_node = self.head
            while actual_node:
                if actual_node.next is None:
                    actual_node.next = Node(iD = iD)
                    break
                actual_node = actual_node.next

    def make_circular(self):
        actual_node = self.head
        i = 0
        while actual_node:
            if actual_node.next is None:
                actual_node.next = self.head
                self.circular_lenght = i
                print(self.circular_lenght)
                break
            actual_node = actual_node.next
            i+=1

    def get_next_n_iDs(self, iD, n):
        actual_node = self.find_iD(iD)
        output = dict()
        for i in range(n):
            output[actual_node.iD] = (i,i+n)
            actual_node = actual_node.next
        return output
    
    def get_next_n_nodes(self, iD, n):
        actual_node = self.find_iD(iD)
        output = []
        for i in range(n):
            output.append(actual_node)
            actual_node = actual_node.next
        return output

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
        
        self.tag_boxes = {}
        self.box_vertices = {}
        self.plane_world_pts = {}
        self.plane_world_pts_detect = []
        self.plane_img_pts_detect = []
        
        self.load_original_points()
        self.compute_plane_dims()
        self.tag_order_linkd_list = LinkedList()
        self.init_tag_boxes()
        self.define_boxes_for_tags()
        self.rotate_original_pts()

        self.camera_matrix = np.loadtxt(calib_path+'camera_matrix.txt', delimiter=',')
        self.camera_distortion = np.loadtxt(calib_path+'distortion.txt', delimiter=',')
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()
    
    def load_original_points(self):
        f = open('plane_points.json')
		# Dict of points in conveyor:
        self.plane_world_pts = json.load(f)
    
    def compute_plane_dims(self):
        
        tl = self.plane_world_pts[self.corners['tl']]
        tr = self.plane_world_pts[self.corners['tr']]
        br = self.plane_world_pts[self.corners['br']]
        bl = self.plane_world_pts[self.corners['bl']]
        
        self.plane_w = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))//10
        self.plane_h = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))//10

        print(self.plane_w, self.plane_h)
    
    def init_tag_boxes(self):
        for iD in self.plane_world_pts:
            self.tag_order_linkd_list.add_end(iD)
            self.tag_boxes[iD] = {'box':None,'pos':None}
        self.tag_order_linkd_list.make_circular()
        print(self.tag_order_linkd_list)

    def compute_tag_relative_pos(self, iD, n):
        actual_node = self.tag_order_linkd_list.find_iD(iD)
        output = dict()
        num_pts = len(self.plane_world_pts)
        for i in range(n):
            output[actual_node.iD] = (i,i+num_pts)
            actual_node = actual_node.next
        return output
    
    def define_template_plane_base(self):
        num_pts = len(self.plane_world_pts)
        plane_base = np.zeros((num_pts,3))

        for i, (key, vector) in enumerate(self.plane_world_pts.items()):
            xyz_pt = np.append(np.array(vector), [0.0])/10
            plane_base[i] = xyz_pt

        self.template_plane_base = plane_base
        print('Template: \n',plane_base)

    def define_boxes_for_tags(self):
        self.define_template_plane_base()
        num_pts = len(self.plane_world_pts)
        
        for i,(iD, vector) in enumerate(self.plane_world_pts.items()):
            xyz_pt = np.append(np.array(vector), [0.0])/10
            xyz_pt_matrix = np.tile(xyz_pt, (num_pts, 1))
            ground_rect = self.template_plane_base - xyz_pt_matrix
            ground_rect_up_crop = ground_rect[:i,:]
            ground_rect_low_crop = ground_rect[i:,:]
            ground_rect = np.concatenate((ground_rect_low_crop, ground_rect_up_crop), axis=0)
            box_3d = np.concatenate((ground_rect, ground_rect), axis=0)
            box_3d[num_pts:,2] = self.box_z
            positions = self.compute_tag_relative_pos(iD, num_pts)
            self.tag_boxes[iD]['box'] = box_3d
            self.tag_boxes[iD]['pos'] = positions

        print(self.tag_boxes)
        
    def rotate_original_pts(self):
        Rot_x = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, math.cos(math.radians(180)),-math.sin(math.radians(180))],
                    [0.0, math.sin(math.radians(180)), math.cos(math.radians(180))]])

        for key, vector in self.plane_world_pts.items():
            xyz_pt = np.append(np.array(vector), [0.0])
            self.plane_world_pts[key] = list(Rot_x @ xyz_pt)[:2]
        print(self.plane_world_pts)

    def compute_homog(self, w_updated_pts = False):
        self.homography = None
        self.plane_img_pts_detect = []
        self.plane_world_pts_detect = []
        box = self.box_verts_update if w_updated_pts else self.box_vertices
        for tag_id in box:
            if tag_id in self.plane_world_pts:
                self.plane_world_pts_detect.append(self.plane_world_pts[tag_id])
                self.plane_img_pts_detect.append(list(box[tag_id][0]))
        is_enough_points_detect = len(self.plane_img_pts_detect)>= 4
        if is_enough_points_detect:
            self.homography,status = cv2.findHomography(np.array(self.plane_img_pts_detect), 
												np.array(self.plane_world_pts_detect))
            return self.homography
        else:
            # print("[INFO]: Less than 4 corresponding points found")
            self.homography = None
            return self.homography
    
    def compute_perspective_trans(self, image, w_updated_pts = False, adaptive_aspect = False):
        height, width, channels = image.shape
        box = self.box_verts_update if w_updated_pts else self.box_vertices
        if self.homography is not None and all(x in box for x in self.corners.values()):
            tl = box[self.corners['tl']][0]
            tr = box[self.corners['tr']][0]
            br = box[self.corners['br']][0]
            bl = box[self.corners['bl']][0]
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
                                        (int(self.plane_w)*10, int(self.plane_h)*10))
            
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
        self.pts_update[pt_idx1] = self.box_vertices[box_vert_id][0]
        self.pts_update[pt_idx2] = self.box_vertices[box_vert_id][1]

    def update_pts_w_id(self, iD):

        for box_vert_id in self.box_vertices:
            if box_vert_id in self.tag_boxes[iD]['pos']:
                self.rewrite_pts(self.tag_boxes[iD]['pos'][box_vert_id], box_vert_id)
            
    def update_img_pts(self, iD, rvec, tvec):
        num_pts = len(self.plane_world_pts)
        self.pts_update = [None] * num_pts * 2
        self.update_pts_w_id(iD)

        world_points = self.define_world_pts(iD)
        img_points, _ = cv2.projectPoints(
                                world_points, 
                                rvec, tvec, 
                                self.camera_matrix, 
                                self.camera_distortion)

        img_points = np.round(img_points).astype(int).reshape(-1, 2)
        # box_verts_update = {'id1':(x1,y1),'id2':(x2,y2),...} in pixels:
        box_verts_update = {
            key:(self.pts_update[i], self.pts_update[i+num_pts]) if self.pts_update[i] 
            else (tuple(img_points[i]),tuple(img_points[i+num_pts])) 
            for i, key in enumerate(self.tag_boxes[iD]['pos'])
            }
        return box_verts_update

    def compute_box_update(self,image, iD, rvec, tvec):
        box_update = self.update_img_pts(str(iD), rvec, tvec)

        cv2.line(image, box_update['0'][0], box_update['1'][0], (255,0,0), 2)
        cv2.line(image, box_update['1'][0], box_update['2'][0], (255,0,0), 2)
        cv2.line(image, box_update['2'][0], box_update['3'][0], (255,0,0), 2)
        cv2.line(image, box_update['3'][0], box_update['0'][0], (255,0,0), 2)
        cv2.line(image, box_update['0'][0], box_update['0'][1], (255,0,0), 2)
        cv2.line(image, box_update['1'][0], box_update['1'][1], (255,0,0), 2)
        cv2.line(image, box_update['2'][0], box_update['2'][1], (255,0,0), 2)
        cv2.line(image, box_update['3'][0], box_update['3'][1], (255,0,0), 2)
        cv2.line(image, box_update['0'][1], box_update['1'][1], (255,0,0), 2)
        cv2.line(image, box_update['1'][1], box_update['2'][1], (255,0,0), 2)
        cv2.line(image, box_update['2'][1], box_update['3'][1], (255,0,0), 2)
        cv2.line(image, box_update['3'][1], box_update['0'][1], (255,0,0), 2)
        return box_update
    
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

            cv2.aruco.drawDetectedMarkers(frame, corners)
            first_id = ids[0][0]
            self.rot_vecs, self.tran_vecs = poses[0], poses[1]
            self.box_vertices = {str(tag_id[0]):self.compute_tag_z_vertices( 
                                                            self.rot_vecs[i][0], 
                                                            self.tran_vecs[i][0]) 
                                                            for i, tag_id in enumerate(ids)}
            for i, tag_id in enumerate(ids):
                
                rvec , tvec = self.rot_vecs[i][0], self.tran_vecs[i][0]
                self.draw_tag_pose(frame, rvec, tvec)

                if tag_id == first_id:

                    self.box_verts_update = self.compute_box_update(
                                                    frame, 
                                                    str(first_id),
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