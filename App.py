import cv2
import os
import datetime as dt
import numpy as np
import tensorflow as tf
import torch
from keras.models import load_model
from ultralytics import YOLO
import cvzone
import math
from twilio.rest import Client
import time

print("Script loaded. Import complete")

# Mohib update the environment variables here
TWILIO_ACCOUNT_SID = 'your_account_sid'
TWILIO_AUTH_TOKEN = 'your_auth_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
RECIPIENT_PHONE_NUMBER = 'recipient_phone_number'

# Seatbelt detection configurations
OBJECT_DETECTION_MODEL_PATH = "./Finding-seatbelt/best.pt"
PREDICTOR_MODEL_PATH = "./Finding-seatbelt/keras_model.h5"
CLASS_NAMES = {0: 'No Seatbelt worn', 1: 'Seatbelt Worn'}
THRESHOLD_SCORE = 0.99
SKIP_FRAMES = 1
MAX_FRAME_RECORD = 500
# INPUT_VIDEO = './Finding-seatbelt/testvideo4.mp4'
# INPUT_VIDEO = './Smoking-detection/cigar1.mp4'
INPUT_VIDEO = 0;
OUTPUT_FILE = 'output/test_result_'+ dt.datetime.strftime(dt.datetime.now(), "%Y%m%d%H%M%S") +'.mp4'
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

def prediction_func(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = (img/127.5)-1
    img = tf.expand_dims(img, axis=0)
    pred = predictor.predict(img)
    index = np.argmax(pred)
    class_name = CLASS_NAMES[index]
    confidence_score = pred[0][index]
    return class_name, confidence_score

def send_sms(message):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )

last_alert_time = time.time()

# Load the seatbelt predictor model
predictor = load_model(PREDICTOR_MODEL_PATH, compile=False)
print("Predictor loaded")

# Load the Ultralytics object detection model
model = torch.hub.load("ultralytics/yolov5", "custom", path=OBJECT_DETECTION_MODEL_PATH, force_reload=False)

# Smoking and drinking detection configurations
smoking_drinking_model = YOLO('./Smoking-detection/best.pt')
classnames=['drinking', 'smoking']

# Video capture
cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

os.makedirs(OUTPUT_FILE.rsplit("/", 1)[0], exist_ok=True)

# Analyzing input video
frame_count = 0
while True:
    ret, img = cap.read()
    if ret:
        frame_count += 1
        if frame_count % SKIP_FRAMES == 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect driver seatbelt
            results = model(img)
            boxes = results.xyxy[0]
            boxes = boxes.cpu()
            for j in boxes:
                x1, y1, x2, y2, score, y_pred = j.numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                img_crop = img[y1:y2, x1:x2]

                y_pred, score = prediction_func(img_crop)

                if y_pred == CLASS_NAMES[0] or Class > 0:
                    current_time = time.time()
                    # Check if 2 hours have passed since the last alert
                    if current_time - last_alert_time >= 7200:  # 7200 seconds = 2 hours
                        # Send alert
                        send_sms("Attention: Seatbelt not worn or smoking/drinking detected!")
                        last_alert_time = current_time  # Update last alert time

                elif y_pred == CLASS_NAMES[1]:
                    draw_color = COLOR_GREEN

                if score >= THRESHOLD_SCORE:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f'{y_pred} {str(score)[:4]}', (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)

            # Detect smoking and drinking
            result = smoking_drinking_model(img, stream=True)
            for info in result:
                boxes = info.boxes
                for box in boxes:
                    connfidence = box.conf[0]
                    connfidence = math.ceil(connfidence * 100)
                    Class = int(box.cls[0])
                    if connfidence >= THRESHOLD_SCORE:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cvzone.putTextRect(img, f'{classnames[Class]} {connfidence}%', [x1+8, y1+100], scale=1.5, thickness=2)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Video feed', img)

    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Script run complete. Results saved to :", OUTPUT_FILE)
