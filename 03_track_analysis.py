import cv2
import numpy as np
from skimage.morphology import thin

# 1. Load image (use local path)
path = 'circuitblacknwhite.png'   # file in same folder as script
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Could not load image at {path}")

# 2. Convert to Black and White
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# 3. Shrink to a 1-pixel line
centerline = thin(binary > 0)

# 4. Save result
result_img = (centerline * 255).astype(np.uint8)
cv2.imwrite('track_centerline.png', result_img)

print("Here is your centerline:")

# 5. Display image (VS Code compatible)
cv2.imshow("Centerline", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()