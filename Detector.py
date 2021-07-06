import cv2 as cv                            # importing all packages
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.models import load_model, Sequential 

face_cas = cv.CascadeClassifier('Haarcascade_frontalface_alt2.xml')    # using Haarcascade Face detection 

from keras.applications.vgg19 import VGG19                # importing and initiating VGG19 CNN Architecture
vg = VGG19(include_top=False, input_shape=(128,128,3))   

model = Sequential()

for layer in vg.layers:
    layer.trainable = False

model.add(vg)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(2, activation="softmax"))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.load_weights('trained_weights.h5')       # Loading trained weights from new_weights.h5 file 


cap = cv.VideoCapture(0)            # Using WebCam 

while True :
    is_true, frame = cap.read()
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cas.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
    cv.putText(frame, 'Press q to exit', (10,30), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (20,20,20), 1, cv.LINE_AA)
    
    for (x, y, w, h) in faces :
        roi = frame[y:y+h, x:x+w]
        font = cv.FONT_HERSHEY_SIMPLEX
        img1 = cv.resize(roi, (128,128))
        img_array = tf.expand_dims(img1, 0)
        
        label = model.predict(img_array)[0][1]           # predicting label for the frame     
        if label >= 0.5 :                                # Labels=={'With_Mask' : [0,1], 'Without_Mask' : [1,0]}
            per = label*100
            label = 1
        else : 
            per = label*100
            label =0
        if(label==0) :
             name='With_Mask :' + str (100-int(per)) + '%'
             color = (0,255,0)
        else : 
             name = 'Without mask :' + str (int(per)) + '%'
             color = (0,0,255)   
        
        stroke = 2
      
        cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
       
        co = (255,0,0) 
        stroke=1
   
        cv.rectangle(frame, (x,y), (x+w,y+h), color, stroke)      
    cv.imshow('Face Mask Detector', frame)

    if cv.waitKey(20) & 0xFF == ord('q') :
        break

cap.release()


cv.destroyAllWindows()