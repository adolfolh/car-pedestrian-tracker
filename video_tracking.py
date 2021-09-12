import cv2

vid_file = 'vid.mp4'

classifier_file = 'cars.xml'

vid = cv2.VideoCapture(vid_file)

car_tracker = cv2.CascadeClassifier(classifier_file)

while True:

    (read_successful_frame, frame) = vid.read()

    if read_successful_frame:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(gray_img)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Tracker, Press 'Q' to Quit", frame)

    # Window waits for 'Q' key to be pressed
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

print("Code Completed")