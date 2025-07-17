import cv2
import numpy as np
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('recep.webm')

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

def get_marker_center(corner):
    pts = corner.reshape((4,2))
    center = pts.mean(axis=0)
    return center

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ret_vid, overlay = video.read()
    if not ret_vid:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_vid, overlay = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()

        selected_corners_23 = []
        centers_23 = []
        selected_corners_49 = []
        centers_49 = []

        for i, id_ in enumerate(ids):
            if id_ == 23:
                selected_corners_23.append(corners[i])
                centers_23.append(get_marker_center(corners[i]))
            elif id_ == 49:
                selected_corners_49.append(corners[i])
                centers_49.append(get_marker_center(corners[i]))

        if len(selected_corners_23) == 2 and len(selected_corners_49) == 2:
            centers_23 = np.array(centers_23)
            idx_left_23 = np.argmin(centers_23[:,0])
            idx_right_23 = np.argmax(centers_23[:,0])

            centers_49 = np.array(centers_49)
            idx_left_49 = np.argmin(centers_49[:,0])
            idx_right_49 = np.argmax(centers_49[:,0])

            pts_dst = np.array([
                selected_corners_23[idx_left_23].reshape(4,2)[0],
                selected_corners_23[idx_right_23].reshape(4,2)[1],
                selected_corners_49[idx_right_49].reshape(4,2)[2],
                selected_corners_49[idx_left_49].reshape(4,2)[3],
            ], dtype=np.float32)

            overlay_height, overlay_width = overlay.shape[:2]
            pts_src = np.array([
                [0, 0],
                [overlay_width - 1, 0],
                [overlay_width - 1, overlay_height - 1],
                [0, overlay_height - 1],
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped_overlay = cv2.warpPerspective(overlay, matrix, (frame.shape[1], frame.shape[0]))

            gray_overlay = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            overlay_fg = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)

            frame = cv2.add(frame_bg, overlay_fg)

    cv2.imshow('AR Table Overlay', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
