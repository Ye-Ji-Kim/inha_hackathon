import cv2

cap = cv2.VideoCapture(0)


while(True):
    isSuccess, frame = cap.read()
    if isSuccess:
        cv2.imshow("My Capture", frame)
        
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    