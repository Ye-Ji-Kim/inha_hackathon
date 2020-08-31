# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:33:47 2020

@author: 융
"""

from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.models import load_model
import os
from PIL import Image

# 카테고리 지정하기
categories = ["no_car","straight","left","right"]
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 64
image_h = 64
# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("./4car.npy",allow_pickle=True)
# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# 모델 구조 정의 

model = Sequential()
print("띠용",X_train.shape[1:])
model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))


# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# 모델 구축하기
model.compile(loss='binary_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])
# 모델 확인
#print(model.summary())

# 모델 훈련하기
#model.fit(X_train, y_train, batch_size=32, nb_epoch=20)
# 학습 완료된 모델 저장
h5_file = "./car-model.h5"
if os.path.exists(h5_file):
    # 기존에 학습된 모델 불러들이기
    model=load_model(h5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    model.fit(X_train, y_train, batch_size=5, nb_epoch=15)
    model.save(h5_file)
    
print("테스트 시작")
#####test
# 적용해볼 이미지  
test_image = ['./33/8-2.png','./33/8-1.png','./11/3-1.png','./11/6-4.png','./22/4-4.png','./22/2-1.png','./00/2.png','./00/1.png']
# 이미지 resize
for i in test_image:
    img = Image.open(i)
    img = img.convert("RGB")
    img = img.resize((64,64))
    data = np.asarray(img)
    X = np.array(data)
    X = X.astype("float") / 256
    X = X.reshape(-1, 64, 64,3)
    # 예측
    pred = model.predict(X)  
    result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
    print('New data category : ',categories[result[0]])