from keras.models import load_model
import cv2
import numpy as np
from collections import deque
model = load_model('devnagiri.h5')
#print(model)
letter_count = ['CHECK','क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','न','प','फ','ब','भ',
              'म','य','र','ल','व','श','ष','स','ह','kshra','tra','gya','CHECK']

def keras_process(img):
    img=cv2.resize(img,(32,32))
    img=np.array(img,dtype=np.float32)
    img=np.reshape(img,(-1,32,32,1))
    return img

def keras_pred(model,image):
    proc=keras_process(image)
    print("Processed : ", proc.shape)
    pred_prob=model.predict(proc)[0]
    pred_cls=list(pred_prob).index(max(pred_prob))
    return max(pred_prob),pred_cls

cap=cv2.VideoCapture(0)
#BGR
Lowerblue=np.array([110,50,50])
Upperblue=np.array([130,255,255])
pred_cls=0
pts=deque(maxlen=512)
blackboard=np.zeros((480,640,3),dtype=np.uint8)
digit=np.zeros((200,200,3),dtype=np.uint8)

while(cap.isOpened()):
    ret,img=cap.read()
    img=cv2.flip(img,1)
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(imgHSV,Lowerblue,Upperblue)
    blur=cv2.medianBlur(mask,15)
    blur=cv2.GaussianBlur(blur,(5,5),0)
    thresh=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cnts=cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
    center=None
    if(len(cnts)>=1):
        contour=max(cnts,key=cv2.contourArea())
        if cv2.contourArea(contour)>250:
            ((x,y),radius)=cv2.minEnclosingCircle(contour)
            cv2.circle(img,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(img,center,5,(0,0,255),-1)
            M=cv2.moments(contour)
            center=(int(M['m10']/M['m00']),int(M['m01'],M['m00']))
            pts.appendleft(center)
            for i in range(1,len(pts)):
                if pts[i-1] is None or pts[i] is None:
                    continue
                cv2.line(blackboard,pts[i-1],pts[i],(255,255,255),10)
                cv2.line(img,pts[i-1],pts[i],(0,0,255),5)
    elif len(cnts)==0:
            if len(pts)!=[]:
                blb_gray=cv2.cvtColor(blackboard,cv2.COLOR_BGR2GRAY)
                blur1 = cv2.medianBlur(blb_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1=cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                blb_cnts=cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1]
                if (len(blb_cnts) >= 1):
                    cnt = max(blb_cnts, key=cv2.contourArea())
                    print(cv2.contourArea(cnt))
                    if cv2.contourArea(cnt) > 2000:
                        x,y,w,h=cv2.boundingRect(cnt)
                        digit=blb_gray[y:y+h,x:x+w]
                        pred_prob,pred_cls=keras_pred(model,digit)
                        print(pred_prob,"\n",letter_count[pred_cls])
            pts=deque(maxlen=512)

            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img,"CONV NETWORK :",letter_count[pred_cls],(10,470),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.imshow("FRAME",img)
    cv2.imshow("Contours",thresh)
    k=cv2.waitKey(10)
    if(k==27):
        break