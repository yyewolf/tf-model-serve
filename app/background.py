import numpy as np
import cv2
import rembg

my_session = rembg.new_session("u2netp")

def remove_background(img):
    img = rembg.remove(img, session=my_session, post_process_mask=True)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    # convert to CV_8UC1
    gray = np.array(gray, dtype=np.uint8)
    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
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