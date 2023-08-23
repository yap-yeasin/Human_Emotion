import numpy as np
import cv2
from deepface import DeepFace

faceModel=cv2.dnn.readNetFromCaffe("mo/opencv_face_detector.prototxt",
caffeModel="mo/res10_300x300_ssd_iter_140000.caffemodel")

# def processVideo(self):
    # cap=cv2.VideoCapture(filename)
cap=cv2.VideoCapture(1) 

if(cap.isOpened()==False):
    print("Error  ")

success,img=cap.read()
_,img_copy = cap.read()

height,width=img.shape[:2]

img = cv2.flip(img, 1)

while success:
    li1 = []
    temp = []
    
    blob=cv2.dnn.blobFromImage(img,1.0,(300,300),(104.0,177.0,123.0),swapRB=False, crop=False)
    faceModel.setInput(blob)

    predictions=faceModel.forward()

    for i in range(0,predictions.shape[2]):
        if predictions[0,0,i,2] > 0.5:
            bbox=predictions[0,0,i,3:7] * np.array([width,height,width,height])
            (xmin,ymin,xmax,ymax)= bbox.astype("int")
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
            # print(xmin,ymin,xmax,ymax )
            temp.append(xmin)
            temp.append(ymin)
            temp.append(xmax)
            temp.append(ymax)
            
            
    li1.append(temp)
    # print(li1)
    cv2.imshow("Output",img)
    
    # print(xmin,ymin,xmax,ymax)
    
    # ## cropped = img[start_row:end_row, start_col:end_col]
    
    crop = img_copy[int(ymin):int(ymax),int(xmin):int(xmax)]
    
    result = DeepFace.analyze(img , actions = ['emotion'])
    
    cv2.putText(img,result['dominant_emotion'],(30,30),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),cv2.LINE_4)
    
    print(result['dominant_emotion'])
    
    cv2.imshow("Crop_Image",crop)
    
    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    key=cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    success,img=cap.read()
    _,img_copy = cap.read()

cap.release()
cv2.destroyAllWindows()
