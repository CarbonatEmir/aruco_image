import numpy as np
import cv2

# 4 kanallı boş resim (şeffaf)
img = np.zeros((200, 200, 4), dtype=np.uint8)

# Kırmızı daire (BGR: 0,0,255), alfa 255 (opak)
cv2.circle(img, (100, 100), 80, (0, 0, 255, 255), -1)

cv2.imwrite('overlay.png', img)
