from ultralytics import YOLO
import cv2
import cvzone

cam = cv2.VideoCapture(0)
cam.set(1,1280)
cam.set(2,720)
model = YOLO('yolov8n.pt')

while True :
    success, img = cam.read()
    results = model(img,stream=True)
    for a in results :
        boxes = a.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3) 
    cv2.imshow("Image",img)
    cv2.waitKey(1)