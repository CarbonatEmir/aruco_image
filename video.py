import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)
video = cv2.VideoCapture('dog.webm')

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ret_vid, overlay = video.read()
    if not ret_vid:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_vid, overlay = video.read()

    overlay_height, overlay_width = overlay.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        for corner in corners:
            pts_dst = corner.astype(np.float32)

            pts_src = np.array([
                [0, 0],
                [overlay_width - 1, 0],
                [overlay_width - 1, overlay_height - 1],
                [0, overlay_height - 1]
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
            warped_overlay = cv2.warpPerspective(overlay, matrix, (frame.shape[1], frame.shape[0]))


            gray_overlay = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            overlay_fg = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)

            frame = cv2.add(frame_bg, overlay_fg)

    cv2.imshow('AR Video Overlay', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
