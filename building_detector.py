from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
img = cv2.imread('D:\Python Programs\object_detection\small area.jpg')
width = 1280
height = 720

resize = cv2.resize(img,(width,height))
result = model(resize,show=True)
cv2.waitKey(0)