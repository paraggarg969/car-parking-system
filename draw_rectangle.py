# draw the rectangle for each parking slot
# 1. find the position where the parking space is available
# 2. preprocessing -> grayscale, gaussian blur, threshold, median blur to remove the noise, dilation

import cv2

image = cv2.imread("Image_Video/image.png")

while True:
    # draw the rectangle around the parking space
    cv2.rectangle(image, (49,145),(156,193),(255,100,100),2)

    cv2.imshow("input imsge", image)
    if cv2.waitKey(1) & 0xFF==ord('1'):
        break