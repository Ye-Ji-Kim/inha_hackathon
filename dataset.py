from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self):
        # 분류 대상 카테고리 선택하기
        self.accident_dir = "./data"
        self.categories = ["None","Straight","Right","Left"]
        self.nb_classes = len(self.categories)
        # 이미지 크기 지정
        self.image_w = 64
        self.image_h = 64
        self.pixels = self.image_w * self.image_h * 3

    def prepare(self):
        # 이미지 데이터 읽어 들이기
        X = []
        Y = []
        for idx, cat in enumerate(self.categories):
            # 레이블 지정
            label = [0 for i in range(self.nb_classes)]
            label[idx] = 1
            # 이미지
            image_dir = self.accident_dir + "/" + cat
            files = glob.glob(image_dir+"/*.jpg")
            for i, f in enumerate(files):
                img = Image.open(f)
                img = img.convert("RGB")
                img = img.resize((self.image_w, self.image_h))
                data = np.asarray(img)      # numpy 배열로 변환
                X.append(data)
                Y.append(label)
                if i % 10 == 0:
                    print(i, "\n", data)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def data_generation(self, X, Y):
        # 학습 전용 데이터와 테스트 전용 데이터 구분
        X_train, X_test, y_train, y_test = \
            train_test_split(X, Y)
        xy = (X_train, X_test, y_train, y_test)

        print('>>> data 저장중 ...')
        np.save("./data/4cars.npy", xy)
        print("ok,", len(Y))

        return xy

    def image_norm(self, train, test):
        # 데이터 정규화하기(0~1사이로)
        train = train.astype("float") / 256
        test = test.astype("float") / 256
        print('X_train shape:', train.shape)

        return train, test
