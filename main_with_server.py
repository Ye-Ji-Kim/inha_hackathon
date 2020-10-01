# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:32:17 2020

@author: kyeoj
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os
import cv2
import serial
from tensorflow.keras.models import load_model
import requests

from model import Model

if __name__ == '__main__':
    M = Model()

    # 카테고리 지정하기
    categories = M.data.categories
    nb_classes = M.data.nb_classes


    #모델 불러오기
    h5_file = "./car-model.h5"
    model = M.load_model(h5_file)

    cap = cv2.VideoCapture(1)
    server = "http://127.0.0.1:5000/"
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
            if result > 0 and result != val:
                result = str(result)
                URL = server + "arduino"
                params = {'result': result}
                res = requests.get(URL, params=params)
                val = result


        if cv2.waitKey(500)&0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()