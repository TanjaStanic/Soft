"""
@author: Tanja Stanic
"""
import cv2
import blue_and_green_lines as find_lines
import numpy as np
import os.path
import help_functions as hf
import matplotlib.pyplot as plt
from keras.models import load_model
from neural_network import neural_network, make_model

# lista sa nazivima videa i
# lista rezultata ciji se sadrzaj zapisuje u out.txt
video_list = []
results_list = []

# funkcija koja poziva ucitavanje svih vedea
def load_all_videos():
    for i in range(10):
        path = 'video-' + str(i) + '.avi'
        print('_____________________________')
        print('___ Video pocinje: ____')
        print(path)
        video_list.append(path)
        # preuzima rezultat sabiranja i oduzimanja       
        # dodaje u listu rezultata
        finall_result = load_video(path)
        print('___ Rezultat videa  ' + str(i) + ' je: ' + str(finall_result) + '____')
        print('____________________________________________________')
        results_list.append(finall_result)
        
# funkcija za ucitavanje jednog videa
def load_video(path):

    num_frame = 0
    video_capture = cv2.VideoCapture(path)
    #indeksiranje frejmova
    video_capture.set(1,num_frame) 
    ret_value, frame = video_capture.read()

    # kad se ucita jedan video, potrebno je samo jednom ocitati koordinate
    # za plavu i zelenu liniju    
    image_frame = hf.open_image(frame)
    blue_lines_coo = find_lines.blue_line_detection(image_frame)
    green_lines_coo = find_lines.green_line_detection(image_frame)

    # ispisuje koordinate linija
    # u formi (x1,y1,x2,y2)
    print(blue_lines_coo)
    print(green_lines_coo)
    
    
    num_of_frames = []
    restart_all = 0
    finall_result = 0
    add_sum = 0
    sub_sum = 0
    
    while True:
        num_frame+=1 
        ret_value, frame = video_capture.read()
        # ako nema povratne vrijednosti(frame nije zahvacen) prekida se
        if not ret_value:
            break
        # uklanjaju se linije na frejmu
        no_line_frame = hf.image_bin_thres(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # selektovanje regiona od interesa
        img_with_rectangles, region_list, coord_list = hf.select_rectangles(frame.copy(), no_line_frame)
        # pronalazenje coskova
        corners = hf.find_corner(coord_list)

        if restart_all == 15:
            restart_all = 0
            num_of_frames = []
        
        it_region=0
        for corner in corners:
            corner1,corner2 = corner
            # detekcija prelaska preko plave linije
            is_blue_line = hf.detect_line_crossing(corner,blue_lines_coo)
            
            # ako je presao preko plave linije
            # provjerava se da li je taj broj vec sabran (vec detektovan)
            if is_blue_line == True:
                num_exist = hf.is_added(corner,num_of_frames)
                if num_exist == False:
                    frame_num = {'map' : (corner1,corner2)}
                    num_of_frames.append(frame_num)
                    num_value = hf.number_detection([region_list[it_region]],alphabet,cnn)
                    print('_______ Detektovano preko plave _______')
                    print(num_value)
                    # prepoznata vrijednost se dodaje u sumu zbira i u ukupnu sumu
                    add_sum+=num_value
                    finall_result+=num_value
            # detekcija prelaska preko zelene linije
            is_green_line = hf.detect_line_crossing(corner,green_lines_coo)
            
            # ako je presao preko zelene linije
            # provjerava se da li je taj broj vec oduzet (vec detektovan)
            if is_green_line == True:
                num_exist = hf.is_added(corner,num_of_frames)
                if num_exist == False:
                    frame_num = {'map' : (corner1,corner2)}
                    num_of_frames.append(frame_num)
                    # [region_list[it_region]] --> it_region iterira za svaki cosak i detektuje se broj
                    # za svaku sliku u listi sortiranih regiona
                    num_value = hf.number_detection([region_list[it_region]],alphabet,cnn)
                    print('_______ Detektovano preko zelene _______')
                    print(num_value)
                    # prepoznata vrijednost se dodaje u sumu razlike i u ukupnu sumu
                    sub_sum+=num_value
                    finall_result-=num_value
            it_region+=1
        restart_all+=1
        
        #icrtavanje
        cv2.putText(img_with_rectangles, 'Zbir plavih: ' + str(add_sum), (10, 25), cv2.FONT_ITALIC, 0.5, (255, 0, 0),1)
        cv2.putText(img_with_rectangles, 'Zbir zelenih: ' + str(sub_sum), (10, 50), cv2.FONT_ITALIC, 0.5, (0, 255, 0),1)
        cv2.putText(img_with_rectangles, 'Konacno:' + str(finall_result), (10, 75), cv2.FONT_ITALIC, 0.5, (0, 0, 255),1)
        #icrtavanje videa sa regionima
        cv2.imshow('Prikaz', img_with_rectangles)
        key = cv2.waitKey(25)
        if key==27:
            break
    cv2.destroyAllWindows()    
    video_capture.release()
    #kad izracunamo rezultat on ce se vracati
    return finall_result

# funkcija za upisivanje rezultata u file
def write_to_file():

    file = open('out.txt', 'w')
    text = 'RA 74/2015 Tanja Stanic\nfile\tsum\n'

    for i in range(10):
        text += video_list[i] + '\t'+ str(results_list[i]) + '\n'

    file.write(text)
    file.close()


# cnn = neural_network()
print('_____ Pokretanje aplikacije _____')
print('_____   Ucitavanje  modela  _____')
cnn = make_model()
cnn.load_weights(''
    'model.h5')
print('_____   Ucitan  model   _____')
alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
load_all_videos()
write_to_file()
print('_____   All  done  _____')