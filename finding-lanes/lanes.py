import cv2
import numpy as np
import matplotlib.pyplot as plt

# First Step: Build Edge Detection Algorithm
# First convert image to gray scale
# Second reduce noise using gausian filter or other filters

# Read image
image = cv2.imread("test_image.jpg")

# Copy Image to a numpy array
lane_image = np.copy(image)


# Canny function
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0) 
    canny = cv2.Canny(blur, 50, 150)
    return canny    
    
def display_lines(image, lines):
    line_image = np.zeros_like(image) # create a black line image
    if lines is not None:
        for line in lines:
            #print(line)
            x1, y1, x2, y2 = line.reshape(4) 
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# Build region of interest meaning a traingle/polygonal shape
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
            [(200, height), (1100, height), (550, 250)]
            ]) # check the image in plot to see the x, y axis value
    mask = np.zeros_like(image) # create a mask using image
    cv2.fillPoly(mask, polygons, 255) # 255 = white polygon
    masked_image = cv2.bitwise_and(image, mask) #apply bitwise and
    return masked_image


# =============================================================================
# # Create named window
# cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)
# 
# 
# canny_image = canny(lane_image)   
# cropped_image = region_of_interest(canny_image)
# 
# # Apply Hough Transformation
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, 
#                         np.array([]), minLineLength=40 , maxLineGap=5) 
# 
# line_image = display_lines(lane_image, lines)
# 
# # blend the processed image with the original colored image
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# 
# cv2.imshow("result", combo_image)
# 
# #plt.imshow(canny)
# #plt.show(canny)
# 
# cv2.waitKey(0)
# cv2.destroyWindow("result")
# =============================================================================

# Capture video
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)   
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, 
                            np.array([]), minLineLength=40 , maxLineGap=5) 
    
    line_image = display_lines(frame, lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()    
cv2.destroyAllWindows()
    