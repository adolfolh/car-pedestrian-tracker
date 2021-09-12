import cv2

img_file = 'img.jpg'

classifier_file = 'cars.xml'

img = cv2.imread(img_file)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

car_tracker = cv2.CascadeClassifier(classifier_file)

cars = car_tracker.detectMultiScale(gray_img)

for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Tracker', img)

cv2.waitKey()

print("Code Completed")