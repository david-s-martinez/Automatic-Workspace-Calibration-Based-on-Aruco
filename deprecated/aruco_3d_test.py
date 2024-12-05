import numpy as np
import cv2
import cv2.aruco as aruco
import sys, time, math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def draw_pose(image, camera_matrix,camera_distortion, marker_size, rvec,tvec, z_rot=-1):
        world_points = np.array([
            4, 0, 0,
            0, 0, 0,
            0, 4, 0,
            0, 0, -4 * z_rot
        ]).reshape(-1, 1, 3) * 0.5 * marker_size

        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, camera_distortion)
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

def draw_grid(image, camera_matrix,camera_distortion, marker_size, rvec, tvec):
        world_points = np.array([
            12.75, 7, 0,
            -12.75, 7, 0,
            -12.75, -7, 0,
            12.75, -7, 0,
            12.75, 7, 3,
            -12.75, 7, 3,
            -12.75, -7, 3,
            12.75, -7, 3,
        ]).reshape(-1, 1, 3) * 0.5 * marker_size

        img_points, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, camera_distortion)
        img_points = np.round(img_points).astype(int)
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
def define_world_pts(iD,grid_w, grid_h, marker_size):
    
    if iD == 0:
        world_points = np.array([
                0, 0, 0,
                grid_w, 0, 0,
                grid_w, -grid_h, 0,
                0, -grid_h, 0,
                0, 0, 3,
                grid_w, 0, 3,
                grid_w, -grid_h, 3,
                0, -grid_h, 3
            ]).reshape(-1, 1, 3) 
        
    if iD == 1:
        world_points = np.array([
                0, 0, 0,
                0, -grid_h, 0,
                -grid_w, -grid_h, 0,
                -grid_w, 0, 0,
                0, 0, 3,
                0, -grid_h, 3,
                -grid_w, -grid_h, 3,
                 -grid_w, 0, 3
            ]).reshape(-1, 1, 3)
        
    if iD == 2:
        world_points = np.array([
                0, 0, 0,
                -grid_w, 0, 0,
                -grid_w, grid_h, 0,
                0, grid_h, 0,
                0, 0, 3,
                -grid_w, 0, 3,
                -grid_w, grid_h, 3,
                0, grid_h, 3,
            ]).reshape(-1, 1, 3)
        
    if iD == 3:
        world_points = np.array([
                0, 0, 0,
                0, grid_h, 0,
                grid_w, grid_h, 0,
                grid_w, 0, 0,
                0, 0, 3,
                0, grid_h, 3,
                grid_w, grid_h, 3,
                grid_w, 0, 3
                
            ]).reshape(-1, 1, 3)
        
    if iD == 4:
        world_points = np.array([
                12.75, 7, 0,
                -12.75, 7, 0,
                -12.75, -7, 0,
                12.75, -7, 0,
                12.75, 7, 3,
                -12.75, 7, 3,
                -12.75, -7, 3,
                12.75, -7, 3,
            ]).reshape(-1, 1, 3)
    
    return world_points * 0.5 * marker_size

def draw_grid_id(image, iD ,camera_matrix,camera_distortion, marker_size, rvec, tvec):
    grid_w = 25.5
    grid_h = 14.0
    world_points = define_world_pts(iD, grid_w, grid_h, marker_size)
    img_points, _ = cv2.projectPoints(world_points, rvec, tvec, camera_matrix, camera_distortion)
    img_points = np.round(img_points).astype(int)
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

id_to_find  = 4
marker_size  = 2 #cm

calib_path = ""
camera_matrix = np.loadtxt(calib_path+'camera_matrix.txt', delimiter=',')
camera_distortion = np.loadtxt(calib_path+'distortion.txt', delimiter=',')

#180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    corners, ids, rejected = cv2.aruco.detectMarkers(image = gray, dictionary = aruco_dict, parameters = parameters,
                              cameraMatrix = camera_matrix, distCoeff = camera_distortion)
    
    if ids is not None and (id_to_find in ids):
        grid_id = ids[0]
        poses = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
        rot_vecs, tran_vecs = poses[0], poses[1]
        # print(rot_vecs,'rot')
        for i, tag_id in enumerate(ids):
            rvec , tvec = rot_vecs[i][0], tran_vecs[i][0]
            cv2.aruco.drawDetectedMarkers(frame, corners)
            # cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)
            # draw_pose(frame, camera_matrix, camera_distortion, marker_size, rvec, tvec)
            # if tag_id == id_to_find:
            if tag_id == grid_id:
                draw_pose(frame, camera_matrix, camera_distortion, marker_size, rvec, tvec)
                # draw_grid(frame, camera_matrix, camera_distortion, marker_size, rvec, tvec)
                # draw_grid_id(frame, id_to_find, camera_matrix, camera_distortion, marker_size, rvec, tvec)
                draw_grid_id(frame, grid_id, camera_matrix, camera_distortion, marker_size, rvec, tvec)
                # Tag position in camera frame
                str_position = "Tag Pos x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
                cv2.putText(frame, str_position, (0, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                # rotation matrix tag to camera
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                R_tc = R_ct.T
                # altitude in terms of euler 321 (Needs to be flipped first)
                roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)
                # Marker's altitude respect to camera frame
                str_attitude = "MARKER Altitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),
                                    math.degrees(yaw_marker))
                cv2.putText(frame, str_attitude, (0, 150), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                #Position and attitude f the camera respect to the marker
                pos_camera = -R_tc*np.matrix(tvec).T
                str_position = "CAMERA Position x=%4.0f  y=%4.0f  z=%4.0f"%(pos_camera[0], pos_camera[1], pos_camera[2])
                cv2.putText(frame, str_position, (0, 200), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
                # altitude of the camera respect to the frame
                roll_camera, pitch_camera, yaw_camera = rotationMatrixToEulerAngles(R_flip*R_tc)
                str_attitude = "CAMERA Altitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_camera),math.degrees(pitch_camera),
                                    math.degrees(yaw_camera))
                cv2.putText(frame, str_attitude, (0, 250), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break