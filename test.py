import numpy as np
import os
import cv2
import serial
from tensorflow.keras.models import load_model
import requests


server = "http://127.0.0.1:5000/"

while(True):
    c = input()
    if(c == 'q'):
        break
    URL = server + "arduino"
    params = {'result': c}
    res = requests.get(URL, params=params)