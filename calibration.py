import cv2
import numpy as np
import glob

checkerboard_size = (7, 6)

objp = np.zeros((checkerboard_size[0]*checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calibration_images/*.jpg')

img_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

print("Kamera matrisi:\n", camera_matrix)
print("Distorsiyon katsayıları:\n", dist_coeffs)

np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)
