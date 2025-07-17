import cv2
import numpy as np
import math

def draw_cube(frame, rvec, tvec, camera_matrix, dist_coeffs, size=0.05):
    points = np.float32([
        [0,0,0],[size,0,0],[size,size,0],[0,size,0],
        [0,0,-size],[size,0,-size],[size,size,-size],[0,size,-size]
    ])
    imgpts, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1,2)
    frame = cv2.drawContours(frame, [imgpts[:4]],-1,(0,255,0),3)
    for i,j in zip(range(4),range(4,8)):
        frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0),3)
    frame = cv2.drawContours(frame, [imgpts[4:]],-1,(0,0,255),3)
    return frame

def rvec_to_euler(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.degrees([x,y,z])

camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

cap = cv2.VideoCapture(0)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict)

marker_length = 0.05

objp = np.array([
    [0, 0, 0],
    [marker_length, 0, 0],
    [marker_length, marker_length, 0],
    [0, marker_length, 0]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        print("image not exist")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for corner in corners:
            try:
                img_points = corner.reshape(-1, 2).astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(objp, img_points, camera_matrix, dist_coeffs)
                if success:
                    frame = draw_cube(frame, rvec, tvec, camera_matrix, dist_coeffs, size=marker_length)
                    euler_angles = rvec_to_euler(rvec)
                    x, y, z = tvec.flatten()
                    cv2.putText(frame, f"Pos: X:{x:.2f} Y:{y:.2f} Z:{z:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Rot: Rx:{euler_angles[0]:.1f} Ry:{euler_angles[1]:.1f} Rz:{euler_angles[2]:.1f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    print("solvePnP failed")
            except Exception as e:
                print("Hata:", e)

    cv2.imshow("Pose Estimation + Cube + Euler", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
