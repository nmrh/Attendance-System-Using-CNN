import os

from keras.models import load_model
import cv2
import numpy as np
from datetime import datetime

model = load_model('model-011.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

#data_path='Training'
#categories=os.listdir(data_path)
#labels=[i for i in range(len(categories))]
#label_dict=dict(zip(categories,labels)) #empty dictionary
#print(labels)
#print(label_dict)

labels_dict={0:'BradPitt',1:'LeoDecap'}
#color_dict={0:(0,255,0)}

def MarkAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



while (True):

    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + w, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        MarkAttendance(labels_dict[label])

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if (key == 27):
        break




cv2.destroyAllWindows()
source.release()
