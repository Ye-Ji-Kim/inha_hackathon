# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 04:55:49 2020

@author: 융
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from dataset import Dataset

import numpy as np
import os
from PIL import Image

def CNN_network(data):
    # 모델 구조 정의
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(data.image_h, data.image_w, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # fully connected layer
    model.add(Flatten())  # 벡터형태로 reshape
    model.add(Dense(512))  # 출력
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(data.nb_classes))
    model.add(Activation('softmax'))

    # 모델 확인
    # print(model.summary())

    return model


class Model:
    def __init__(self):
        self.data = Dataset()
        self.network = CNN_network(self.data)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_model(self, model_dir):
        if os.path.exists(model_dir):
            # 기존에 학습된 모델 불러들이기
            self.network = load_model(model_dir)

        else:
            # 학습한 모델이 없으면 학습 후 모델 파일로 저장
            self.fit()

        return self.network

    def fit(self, saved_data = True):
        if saved_data:
            # 데이터 열기
            self.X_train, self.X_test, self.y_train, self.y_test = np.load("./data/4cars.npy")
        else:
            X, Y = self.data.prepare()
            self.X_train, self.X_test, self.y_train, self.y_test = self.data.data_generation(X, Y)

        self.X_train, self.X_test = self.data.image_norm(self.X_train, self.X_test)

        # 모델 구축하기
        self.network.compile(loss='binary_crossentropy',   # 최적화 함수 지정
            optimizer='adam',
            metrics=['accuracy'])

        # 모델 훈련하기
        self.network.fit(self.X_train, self.y_train, batch_size=32, nb_epoch=20)

        # 학습 완료된 모델 저장
        h5_file = "./car-model.h5"
        self.network.model.save(h5_file)



    def eval(self):
        # 모델 평가하기
        score = self.network.evaluate(self.X_test, self.y_test)
        print('loss=', score[0])        # loss
        print('accuracy=', score[1])    # acc

        # 적용해볼 이미지
        test_image = ['./data/None/0.jpg','./data/Straight/1.jpg','./data/Right/7.jpg','./data/Left/7.jpg']
        # 이미지 resize
        for i in test_image:
            img = Image.open(i)
            img = img.convert("RGB")
            img = img.resize((64,64))
            data = np.asarray(img)
            X = np.array(data)
            X = X.astype("float") / 256
            X = X.reshape(-1, 64, 64, 3)
            # 예측
            pred = self.network.predict(X)
            result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
            print('New data category : ',self.data.categories[result[0]])