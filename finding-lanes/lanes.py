import cv2
import numpy as np
import matplotlib.pyplot as plt

# First Step: Build Ednge Detection Algorithm
# First convert image to gray scale
# Second reduce noise using gausian filter or other filters

# Read image
image = cv2.imread("test_image.jpg")

# Copy Image to a numpy array
lane_image = np.copy(image)


# Canny function
def canny(image):
    # Step 1: 
    # Create a grayscale image
    # Why? 
    # 1. gray scale image has single channel where 
    # colored image has more at least 3 channels, 
    # So, gray scale computation is fast and less computing power needed. 
    # And, we will detect lane lines using black and white color values
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    
    
    # Step 2: Reduce noise - Apply Guasian Filter
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    
    # Find the strongest gradients of the image
    canny = cv2.Canny(blur, 50, 150)
    return canny    
    
 
canny = canny(lane_image)   

 
# Show in a plot
plt.imshow(canny)
plt.show()

