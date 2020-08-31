from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os
import cv2
import serial
from tensorflow.keras.models import load_model

PORT = 'COM3'
BaudRate = 9600

ARD=serial.Serial(PORT, BaudRate)

# 카테고리 지정하기
categories = ["None","Straight","Right","Left"]
nb_classes = len(categories)


#모델 불러오기
h5_file = "./car-model.h5"
model = load_model(h5_file)

cap = cv2.VideoCapture(1)
cnt = 0
val = 0

while cap.isOpened():
    isSuccess,frame = cap.read()
    if isSuccess:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(64,64))
        X = np.array(img)
        X = X.astype("float")/256
        X = X.reshape(-1, 64, 64, 3)
        
        pred = model.predict(X)
        result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
        
        result = result[0]
     
        if result > 0 and result > val:
            val = result
            result = str(result)
            print(result)
            result = result.encode('utf-8')
            ARD.write(result)
            

    if cv2.waitKey(500)&0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    