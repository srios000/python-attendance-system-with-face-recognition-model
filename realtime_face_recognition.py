#!/usr/bin/env python
# coding: utf-8

import cv2, numpy as np, face_recognition, os, tensorflow as tf#, time
from mtcnn.mtcnn import MTCNN
from datetime import datetime

tf.compat.v1.disable_eager_execution()

path = 'img\samples'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

#function to compute encodings
def findEncodings(images):
    encodeList = [] #list that will have all encodings
    #convert images to RGB
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
        print(encodeList)
    return encodeList

encoded_face_train = findEncodings(images)

print('Proses Encoding Selesai')
def markAttendance(名前, 性別, 年齢層):
    #open Attendance.csv, read and write at the same time
    with open('Attendance.csv','r+', encoding='windows-1252') as f:
        myDataList = f.readlines()
        nameList = []
        if 名前 not in nameList:
            now = datetime.now()
            dtString = now.strftime('%m/%d/%Y,%H:%M:%S')
            f.writelines(f'\n{名前},{dtString},{性別}, {年齢層} yrs')
            for line in myDataList:
                entry = line.split(',',1)
                nameList.append(entry[0])

realtime_cam = cv2.VideoCapture(0) #initialize the webcam
mtcnn_detector = MTCNN()
all_face_locations = []
if (realtime_cam.isOpened() == False):
    print("Error opening video stream or file")
#loop through every frame in the video
while True:
    success, img = realtime_cam.read()
    s_img = cv2.resize(img,(0,0),None,0.25,0.25) #reduce image size to 0.25 in x and y scale
    s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
    all_face_locations = mtcnn_detector.detect_faces(s_img) #detect all face in camera
    print(all_face_locations)
    
    facesCurFrame = face_recognition.face_locations(s_img, model="hog")
    encodesCurFrame = face_recognition.face_encodings(s_img,facesCurFrame)

    for index, current_face_location in enumerate(all_face_locations):
        x,y,width,height = current_face_location['box']
        left_pos = x
        top_pos = y
        right_pos = x+width
        bottom_pos = y+height
        #change the position maginitude to fit the actual size video frame
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        current_face_image = img[top_pos:bottom_pos,left_pos:right_pos]
        current_face_image = img[top_pos:bottom_pos,left_pos:right_pos]
       
        AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

        current_face_image_blob = cv2.dnn.blobFromImage(current_face_image, 1, (227,227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB = False)
        
        #性別
        gender_label_list = ['L', 'P']
        gender_protext = "dataset/gender_deploy.prototxt"
        gender_caffemodel = "dataset/gender_net.caffemodel"
        gender_cov_net = cv2.dnn.readNet(gender_caffemodel, gender_protext)
        gender_cov_net.setInput(current_face_image_blob)
        gender_predictions = gender_cov_net.forward()
        性別 = gender_label_list[gender_predictions[0].argmax()]

        #年齢層
        age_label_list = ['0 - 2', '4 - 6', '8 - 12', '15 - 20', '25 - 32', '38 - 43', '48 - 53', '60 - 100']
        age_protext = "dataset/age_deploy.prototxt"
        age_caffemodel = "dataset/age_net.caffemodel"
        age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)
        age_cov_net.setInput(current_face_image_blob)
        age_predictions = age_cov_net.forward()
        年齢層 = age_label_list[age_predictions[0].argmax()]
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 年齢層+" "+"yrs", (left_pos, bottom_pos), font, 0.5, (0,255,255),1)
    #compare faces in realtime_cam with all faces in the folder.
    for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
        # matches = face_recognition.compare_faces(encoded_face_train,encodeFace)
        faceDis = face_recognition.face_distance(encoded_face_train,encodeFace)
        os.system('CLS')
        matchIndex = np.argmin(faceDis) #find the lowest elements in  the list to find the best match
        print('Face Distance antara wajah di kamera dengan wajah yang paling mirip di database : ', faceDis[matchIndex])

        if faceDis[matchIndex] < 0.40:
            recognized = 'img/recognized/'
            名前 = classNames[matchIndex].upper().lower()
            fileName2 = recognized+名前
            if os.path.exists(recognized) == False:
                os.mkdir(recognized)
            if os.path.exists(fileName2) == False:
                os.mkdir(fileName2)
            cv2.imwrite('img/recognized/'+名前+'/'+名前+'.jpg', img)
            print('You are identified by name: ', 名前, ', ', 性別, ', 年齢：', 年齢層, '歳. \nYour first data has been saved to the attendance.csv dataframe.') 
        else: 
            名前 = 'Unknown'
            print('Your face is ', 名前, ', not recognized.\nYour data can not be saved to attendance.csv dataframe.') 
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,名前,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

        
    
    #display the video
    cv2.imshow("Webcam Video",img)
    
    k = cv2.waitKey(1)  # jika menekan tombol esc akan berhenti
    if k == 27: 
        break



markAttendance(名前, 性別, 年齢層)
#release the stream and cam
#close all opencv windows open
realtime_cam.release()
cv2.destroyAllWindows()
# Audio(sound_file, autoplay=False)
