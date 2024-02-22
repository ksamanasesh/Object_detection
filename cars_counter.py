import cv2

get_vid = cv2.VideoCapture('cars2.mp4')

car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    isTrue, cap = get_vid.read()
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    car_detect = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    print(f'cars={len(car_detect)}')
    for (x,y,w,h) in car_detect:
        cv2.rectangle(cap, (x,y), ((x+w),(y+h)), (0,0,255),thickness=2)
    
    cv2.imshow('Car Counter', cap)
    if cv2.waitKey(1)==27:

        break

get_vid.release()


