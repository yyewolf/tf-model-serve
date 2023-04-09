import numpy as np
import cv2

def perfect_crop(img):
    # center is white, background is transparent
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)[1]
    trans_mask = thresh[:,:,3] == 0
    thresh[trans_mask] = [255, 255, 255, 255]
    bgr = cv2.cvtColor(thresh, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # erode the image
    eroded = cv2.dilate(cv2.erode(gray, np.ones((5,5), np.uint8)), np.ones((5,5), np.uint8))
    contours = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    extreme_points = np.array([contours[0][0][0], contours[0][0][0], contours[0][0][0], contours[0][0][0]])

    for contour in contours:
        contour = np.squeeze(contour, axis=1)
        contour_min_x, contour_min_y = np.min(contour, axis=0)
        contour_max_x, contour_max_y = np.max(contour, axis=0)
        
        extreme_points[0] = np.minimum(extreme_points[0], [contour_min_x, contour_min_y])
        extreme_points[1] = np.maximum(extreme_points[1], [contour_max_x, contour_max_y])
        extreme_points[2] = np.minimum(extreme_points[2], [contour_min_x, contour_min_y])
        extreme_points[3] = np.maximum(extreme_points[3], [contour_max_x, contour_max_y])
    # crop 
    cropped = img[extreme_points[0][1]:extreme_points[1][1], extreme_points[2][0]:extreme_points[3][0]]
    return cropped