import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Alfa kanallı overlay dosyasını yükle
overlay = cv2.imread('overlay.png', cv2.IMREAD_UNCHANGED)
overlay_height, overlay_width = overlay.shape[:2]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    print("Detected IDs:", ids)

    if ids is not None:
        # Marker köşelerine kırmızı çizgi çiz
        for corner in corners:
            pts = corner.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[0][i]), tuple(pts[0][(i+1) % 4]), (0, 0, 255), 2)

        # Sadece ID 23 olan marker için overlay bindir
        for i, corner in enumerate(corners):
            if ids[i][0] == 23:
                pts_dst = corner.astype(np.float32)

                pts_src = np.array([
                    [0, 0],
                    [overlay_width - 1, 0],
                    [overlay_width - 1, overlay_height - 1],
                    [0, overlay_height - 1]
                ], dtype=np.float32)

                matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
                warped_overlay = cv2.warpPerspective(overlay, matrix, (frame.shape[1], frame.shape[0]))

                alpha_mask = warped_overlay[:, :, 3]
                alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])
                alpha_mask = alpha_mask.astype(float) / 255.0
                alpha_inv = 1.0 - alpha_mask

                overlay_rgb = warped_overlay[:, :, :3].astype(float)
                frame_rgb = frame.astype(float)

                frame = (alpha_inv * frame_rgb + alpha_mask * overlay_rgb).astype(np.uint8)

    cv2.imshow('AR Overlay with Transparency', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
