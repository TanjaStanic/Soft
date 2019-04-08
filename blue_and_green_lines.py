"""
@author: Tanja Stanić
"""
import cv2
import math
import help_functions as hf

def blue_line_detection(input_picture_frame):

    blue_frame = input_picture_frame.copy();
    blue_frame[:,:,1] = 0
    
    hsv = hf.image_to_gray(blue_frame)
    
    ret, th = cv2.threshold(hsv, 20, 255, cv2.THRESH_BINARY)
    
    blue_lines = hf.color_lines(th)

    length = 0
    x1 = min(blue_lines[:, 0, 0])
    y1 = max(blue_lines[:, 0, 1])
    x2 = max(blue_lines[:, 0, 2])
    y2 = min(blue_lines[:, 0, 3])
    
    ret_val = (0,0,0,0)
   
    if not (blue_lines is None):
        for index,l in enumerate(blue_lines):
          
            blue_length = math.sqrt((x2 -x1) ** 2
                                 + (y2 - y1) ** 2)
            if length < blue_length:
                ret_val = (x1,y1,x2,y2)
                length = blue_length
                #print(blue_edges)
            return ret_val

#detekcija zelene linije linije
def green_line_detection(input_picture_frame):

    green_frame = input_picture_frame.copy();
    green_frame[:,:,0] = 0
    
    hsv = hf.image_to_gray(green_frame)
    ret, th = cv2.threshold(hsv, 20, 255, cv2.THRESH_BINARY)
    
    green_lines = hf.color_lines(th)
    length = 0

    x1 = min(green_lines[:, 0, 0])
    y1 = max(green_lines[:, 0, 1])
    x2 = max(green_lines[:, 0, 2])
    y2 = min(green_lines[:, 0, 3])
    
    ret_val = (0,0,0,0)

    if not (green_lines is None):
        for index,l in enumerate(green_lines):
          
            green_length = math.sqrt((x2 -x1) ** 2 + (y2 - y1) ** 2)
            if length < green_length:
                length = green_length
                ret_val = (x1,y1,x2,y2)
                #print(green_edges)
            return ret_val