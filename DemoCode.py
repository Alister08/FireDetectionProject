from ultralytics import YOLO
import cvzone
import cv2
import math
import time
cap = cv2.VideoCapture("fire2.mp4")
model = YOLO('best.pt')
classnames = ['fire']

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))
    result = model(frame, stream=True)
    
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                  scale=1.5, thickness=2)  
                 # Save the IP address and time to a txt file
#                 with open('detected_camera.txt', 'a') as file:
#                     current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#                     file.write(f"Camera IP fire detected:, Detected at: {current_time}\n")
    
    cv2.imshow('frame', frame)
    
    # Calculate the delay based on the video's frame rate
    delay = int(1000 / fps)
    
    # Wait for 'delay' milliseconds. This will ensure the video is displayed at its actual speed.
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
