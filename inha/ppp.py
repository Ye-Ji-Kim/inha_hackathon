# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:25:13 2020

@author: 융
"""

from PIL import Image
import os, glob
import numpy as np
from numpy import random as np_random
from sklearn.model_selection import train_test_split

# 분류 대상 카테고리 선택하기 
accident_dir = "."
categories = ["00","11","22","33"]
nb_classes = len(categories)
# 이미지 크기 지정 
image_w = 64 
image_h = 64
pixels = image_w * image_h * 3
# 이미지 데이터 읽어 들이기 
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정 
    print(idx)
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = accident_dir + "/" + cat
    print(image_dir)
    files = glob.glob(image_dir+"/*.png")
    
    for i, f in enumerate(files):
        print(i)
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        X.append(data)
        Y.append(label)
        if i % 2 == 0:
            print(i, "\n", data)
X = np.array(X)
Y = np.array(Y)
# 학습 전용 데이터와 테스트 전용 데이터 구분 
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y,test_size=0.05)
xy = (X_train, X_test, y_train, y_test)

print('>>> data 저장중 ...')
np.save("./4car.npy", xy)
print("ok,", len(Y))