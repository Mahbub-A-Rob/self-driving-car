import cv2
import numpy as np


# First Step: Build Ednge Detection Algorithm
# First convert image to gray scale
# Second reduce noise using gausian filter or other filters

# Read image
image = cv2.imread("test_image.jpg")

# Copy Image to a numpy array
lane_image = np.copy(image)

# Step 1: 
# Create a grayscale image
# Why? We will detect lane lines using black and white color values
gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)


# Step 2: Reduce noise - Apply Guasian Filter
blur = cv2.GaussianBlur(gray, (5, 5), 0) 

# Use gradient of the image
canny = cv2.Canny(blur, 50, 150)

# Show Image
cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
cv2.imshow("result", canny)
cv2.waitKey(0)
cv2.destroyWindow("result")