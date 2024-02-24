from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real time from webcam
cap = cv2.VideoCapture('./test_video2.mp4')
model = YOLO('./best.pt')

# Reading the classes
classnames=['drinking', 'smoking']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            connfidence = box.conf[0]
            connfidence = math.ceil(connfidence * 100)
            Class = int(box.cls[0])
            if connfidence > 50:
                x1,y1,x2,y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {connfidence}%', [x1+8, y1+100], scale=1.5, thickness=2)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)















# import cv2

# cap = cv2.VideoCapture("./Smoking_detection.mp4")

# # Object detection from a stable camera
# object_detector = cv2.createBackgroundSubtractorMOG2()

# while True:
#     ret, frame = cap.read()

#     # Check if the frame is empty
#     if not ret:
#         break

#     # Increase video size
#     frame = cv2.resize(frame, (1280, 720))

#     # Convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Extract Region of Interest (ROI)
#     roi = gray_frame[100:720, 0:800]

#     # 1. Object Detection
#     mask = object_detector.apply(roi)
#     _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         # Calculate area and remove small elements
#         area = cv2.contourArea(cnt)
#         if area > 100:
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display the frames
#     cv2.imshow('Original Video', frame)
#     cv2.imshow('Grayscale ROI with Object Detection', roi)

#     key = cv2.waitKey(30)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2

# # Use the default webcam (0) or specify another index if you have multiple webcams
# cap = cv2.VideoCapture(0)

# # Object detection from a stable camera
# object_detector = cv2.createBackgroundSubtractorMOG2()

# while True:
#     ret, frame = cap.read()

#     # Convert to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Extract Region of Interest (ROI)
#     roi = gray_frame[200:720, 0:800]

#     # 1. Object Detection
#     mask = object_detector.apply(roi)
#     _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         # Calculate area and remove small elements
#         area = cv2.contourArea(cnt)
#         if area > 100:
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Display the frames
#     cv2.imshow('Webcam Feed', frame)
#     cv2.imshow('Grayscale ROI with Object Detection', roi)

#     key = cv2.waitKey(30)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

