import face_recognition
import cv2
import numpy as np
import csv
import os 
import time
from datetime import datetime

def encode(name):
        name_image = face_recognition.load_image_file(f"C:/work/python projects/working/Face_Recognition/photos/{name}")
        name_encoding = face_recognition.face_encodings(name_image)[0]
        list_from_numpy = list()
        for items in name_encoding:
            list_from_numpy.append(items)
        return list_from_numpy

def face_rec():
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_locations = []
    face_encodings = []
    face_names = []
    s=True

    current_date =datetime.now().strftime("%Y-%m-%d")
    
    try:
        f = open(current_date+'.csv','a',newline = '')
        lnwriter = csv.writer(f)
    except:
        f = open(current_date+'.csv','w+',newline = '')
        lnwriter = csv.writer(f)

    known_face_encoding=[]
    known_faces_names=[]
    names_list=os.listdir('C:/work/python projects/working/Face_Recognition/photos')

    for name in names_list:
        enc=encode(name)
        known_face_encoding.append(enc)

    for name in names_list:
        a=name[-5::-1]
        b=a[::-1]
        known_faces_names.append(b)

    while True:
        if len(known_faces_names)==0:
            break
        _,frame = video_capture.read()
        small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
                name=""
                face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_faces_names[best_match_index]
                    current_date =datetime.now().strftime("%Y-%m-%d")
                    present_time =datetime.now().strftime("%H.%M.%S")
                
                face_names.append(name)
                if name in known_faces_names:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10,100)
                    fontScale              = 1.5
                    fontColor              = (255,0,0)
                    thickness              = 3
                    lineType               = 2
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    cv2.putText(frame,name+' Present', 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
                        
                    if name in known_faces_names:
                        print(f"{name} is present\t\t",current_date,present_time)
                        lnwriter.writerow([name,current_date,present_time])
                        index=known_faces_names.index(name)
                        known_face_encoding.pop(index)
                        known_faces_names.remove(name)
                    
                        
        cv2.imshow("attendence system",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
    f.close()
    return 1

def Input():
        n1=int(input("Enter the number of persons: "))
        cam = cv2.VideoCapture(0)
        for i in range(n1):
                inp = input('Enter person name: ')
                n=2
                while(1): 
                        result,image = cam.read()
                        cv2.imshow(inp,image)
                        if cv2.waitKey(1):
                                path = 'C:/work/python projects/working/Face_Recognition/photos'
                                cv2.imwrite(os.path.join(path ,inp+".jpg"), image)
                                time.sleep(2)
                                n=n-1
                        if n==0:
                                break
        print("Image Taken")
        return 1

