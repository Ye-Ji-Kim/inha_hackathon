# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:39:20 2020

@author: kyeoj
"""

import cv2

cap = cv2.VideoCapture(1)
cnt0 = 69
cnt1 = 77
cnt2 = 95
cnt3 = 64

while(True):
    isSuccess, frame = cap.read()
    if isSuccess:
        cv2.imshow("My Capture", frame)
    '''  
    if cv2.waitKey(1) & 0xFF == ord('n'):
        root0 = './data/None/'+str(cnt0)+'.jpg'
        cv2.imwrite(root0, frame)
        cnt0+=1
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        root1 = './data/Straight/'+str(cnt1)+'.jpg'
        cv2.imwrite(root1, frame)
        cnt1+=1
       
    if cv2.waitKey(1) & 0xFF == ord('r'):
        root2 = './data/Right/'+str(cnt2)+'.jpg'
        cv2.imwrite(root2, frame)
        cnt2+=1
    '''    
    if cv2.waitKey(1) & 0xFF == ord('l'):
        root3 = './data/Left/'+str(cnt3)+'.jpg'
        cv2.imwrite(root3, frame)
        cnt3+=1
       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    