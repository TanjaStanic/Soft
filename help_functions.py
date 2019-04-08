# -*- coding: utf-8 -*-
"""
@author: Tanja Stanic
"""
import cv2
import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils

def image_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# skalira vrijednosti 0-255 na brojeve 0 i 1
def image_scale(image):
    return image/255

# dio slike 28x28 skaliramo u vektor od 784 el
def image_to_vector(image):
    return image.flatten()

def image_bin_thres(img):
    height, width = img.shape[0:2]
    ret, image_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def color_lines(th):
    maxLineGap=10
    minLineLength=100
    return cv2.HoughLinesP(th, 1, np.pi / 180, 100, minLineLength, maxLineGap)

def image_erode(image):
    kernel = np.ones((3,3)) 
    return cv2.erode(image, kernel, iterations=1)

def image_dilate(image):
    kernel = np.ones((3,3))
    return cv2.dilate(image, kernel, iterations=1)

# funkcija koja vrsi erode i delate prilikom otvaranja utvaranja(prikazivanja) slike
def open_image(image_frame):
    img_erode = image_erode(image_frame)
    return_image = image_dilate(img_erode)
    return return_image

# funkcija koje ima ulazni parametar neki region
# i prosiruje taj region na dimenzije 28x28    
def make_28x28_region(area):
    return cv2.resize(area, (28, 28), interpolation=cv2.INTER_NEAREST)

# funkcija koja vrsi selekciju regiona od interesa
# za oznacavanje regiona koristi se funkcija boundingRect
# povratna vrijednost je orginalna slika sa oznacenim regionima sortiranim po x osi
def select_rectangles(img_orig, img_bin):
    
    #sortirani regioni po x osi
    x_axis_sorted_regions = []
    #pomocna lista nesortiranih regiona
    regions_list = []
    #lista koordinata
    coord_list = [];
    
    image, contours, hierarchy = cv2.findContours(img_bin.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        #preuzimanje koordinata
        x_axis, y_axis, width, height = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        if (area>15 and height<45 and height>13 and width>2) or (area>15 and height<45 and height>9 and width>12):
            # sa binarne slike se korpiraju naredni piksele i smijestamo u novu sliku (u nas region)
            region = img_bin[y_axis:y_axis+height+1, x_axis:x_axis+width +height]
            # dodajemo u pomocnu listu regiona 
            # a da prije toga taj region prosirimo na dimenzija 28x28
            regions_list.append([make_28x28_region(region),(x_axis,y_axis,width,height)])
            coord_list.append([x_axis,y_axis,width,height])
            # na orginalnoj slici se oznaca region pravougaonikom
            cv2.rectangle(img_orig, (x_axis, y_axis), (x_axis + width, y_axis + height), (0, 255, 0), 2)
           
    regions_list = sorted(regions_list, key=lambda item:item[1][0])
    x_axis_sorted_regions = x_axis_sorted_regions = [region[0] for region in regions_list]
    coord_list = sorted(coord_list, key=lambda item:item[0])
    return img_orig,x_axis_sorted_regions, coord_list
    

# pronalazenje coska u selektovanom pravougaoniku
# kako bi se obradila samo informacija kad je taj cosak presao liniju
def find_corner(corners):
    ret_val = []
    for c in corners:
        x,y,w,h = c
        ret_val.append([x+w,y+h]) #donji desni dio  ugao
    return ret_val

#----------------------------------------------------------------------------------------------------------
# funkcije za prepoznavanje detektovanog broja 
# ulazne vrijednosti su: slika(region) broja, alfabet, i model nm
# povratna vrijednost je detektovan broj
def number_detection(img,alphabet,cnn):
    # sliku prebacimo u niz
    img_array = np.array(img)
    # zatim skaliramo od 0-1 i prebacimo u matricu sa 4 dimenzije da bi odgovarala ulazu u neuronsku
    in_matrix = scale_array(img_array)
    ret_val = cnn.predict(np.array(in_matrix,np.float32))
    ret_value = show_detected_number(alphabet,ret_val)[0]
    return ret_value

# funkcija u kojoj se skaliraju elementi na 0-1    
def scale_array(img_array):
    in_vector = []
    for img in img_array:
        # elementi se skaliraju
        element = image_scale(img)
        # skalirani elementi se dodaju u input vector
        in_vector.append(element)
    
    in_vector = np.expand_dims(in_vector, axis=3)
    return in_vector
        
# funkcija koja prikazuje detektovan broj
# za svaki rezultat se pronadje indeks odgovarajuceg regiona koji je i indeks u alfabetu
# u rezultat se dodaje karakter iz alfabeta
def show_detected_number(alphabet, values):
    ret_val = []
    for v in values:
        ret_val.append(alphabet[stimulated_neuron(v)])
    return ret_val

# funkcija koja vraca najvise pobudjen neuron
def stimulated_neuron(n):
     return max(enumerate(n), key=lambda x: x[1])[0]

#----------------------------------------------------------------------------------------------#
# funkcije za detekciju prelazenja linije n
# na osnovu coska broja i koordinata linije  
# izracunavaju se k,n iz jednacine prave pomocu funkcije za izracunavanje jednacine (polyfit)
def detect_line_crossing(corner,coordinates):     
    x1,y1,x2,y2 = coordinates
    corner1,corner2 = corner
    x = [x1,x2]
    y = [y1,y2]
    temp = np.polyfit(x,y,1)
    k = temp[0] 
    n = temp[1]
     
    if x2+2>=corner1>=x1-5 and y1+5>=corner2>=y2-1:
        line = k*corner1+n
        if abs(int(corner2)-int(line))<=2:
            return True
    return False

# funkcija koja provjerava da li je broj vec sabran/oduzet (vec detektovan)
# na osnovu coska tog broja i broja prikazanih frejmova
def is_added(corner,num_of_frames):
    corner1,corner2 = corner
    num = {'map' : (corner1,corner2)}
    num_of_exist = num_exist(num,num_of_frames)
    exist_size = len(num_of_exist)
    if exist_size==0:
        return False
    else:
        return True
    
# racuna distancu izmedju trenutno posmatranog broja i brojeva u frejmu
# ako je distanca mala onda se dodaje u brojeve frejma
def num_exist(num,num_of_frames):
    ret_val_num = []
    for n in num_of_frames:
        len = distance(num['map'],n['map'])
        if len < 10:
            ret_val_num.append(n)
    return ret_val_num

def distance(a,b):
    temp_vector = vector(a,b)
    x,y = temp_vector
    return math.sqrt(x * x + y * y)

def vector(b, e):
    x, y = b
    X, Y = e
    return X-x, Y-y