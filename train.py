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

import numpy as np
import os
from PIL import Image

# 카테고리 지정하기
categories = ["None","Straight","Right","Left"]
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 64
image_h = 64
# 데이터 열기 
X_train, X_test, y_train, y_test = np.load("./data/4cars.npy")
# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# 모델 구조 정의 
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


#fully connected layer
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 모델 구축하기
model.compile(loss='binary_crossentropy',   # 최적화 함수 지정
    optimizer='adam',
    metrics=['accuracy'])
# 모델 확인
#print(model.summary())

# 모델 훈련하기
#model.fit(X_train, y_train, batch_size=32, nb_epoch=20)
# 학습 완료된 모델 저장
h5_file = "./car-model.h5"
if os.path.exists(h5_file):
    # 기존에 학습된 모델 불러들이기
    model = load_model(h5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

    model.fit(X_train, y_train, batch_size=32, nb_epoch=20,callbacks=[es])
    model.save(h5_file)
    
# 모델 평가하기 
score = model.evaluate(X_test, y_test)
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
    X = X.reshape(-1, 64, 64,3)
    # 예측
    pred = model.predict(X)  
    result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
    print('New data category : ',categories[result[0]])