import cv2
import os



cap = cv2.VideoCapture(0)
count = 0

cv2.namedWindow('Calibration Capture')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Calibration Capture', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = f'calibration_images/img_{count:02d}.jpg'
        cv2.imwrite(filename, frame)
        print(f"kaydedildi {filename}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
