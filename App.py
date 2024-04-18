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
# from twilio.rest import Client
import time

print("Script loaded. Import complete")

# Hamza update the environment variables here
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
INPUT_VIDEO = './mohibtemp.mp4'
OUTPUT_FILE = 'output/test_result_'+ dt.datetime.strftime(dt.datetime.now(), "%Y%m%d%H%M%S") +'.mp4'
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)

output_width = 1000
output_height = 600

def log_activity(activity):
    with open('activity_log.txt', 'a') as file:
        file.write(activity + '\n')

def prediction_func(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = (img/127.5)-1
    img = tf.expand_dims(img, axis=0)
    pred = predictor.predict(img)
    index = np.argmax(pred)
    class_name = CLASS_NAMES[index]
    confidence_score = pred[0][index]
    return class_name, confidence_score

# def send_sms(message):
#     client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
#     client.messages.create(
#         body=message,
#         from_=TWILIO_PHONE_NUMBER,
#         to=RECIPIENT_PHONE_NUMBER
#     )

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

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 20.0, (output_width, output_height))

def draw_dashboard(img, smoking_detected, drinking_detected, seatbelt_detected):
    # Draw a rectangle for the dashboard
    dashboard_color = (0, 0, 0)  # Black color for the dashboard
    dashboard_height = 100
    cv2.rectangle(img, (0, 0), (output_width, dashboard_height), dashboard_color, -1)
    cv2.putText(img, "Dashboard", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw indicators for smoking, drinking, and seatbelt
    indicator_size = 20
    indicator_padding = 10
    indicator_start_x = 150
    cv2.putText(img, "Smoking:", (indicator_start_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "Drinking:", (indicator_start_x, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(img, "Seatbelt:", (indicator_start_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.rectangle(img, (indicator_start_x + 70, 35), (indicator_start_x + 70 + indicator_size, 35 + indicator_size), (0, 255, 0) if smoking_detected else (0, 0, 255), -1)
    cv2.rectangle(img, (indicator_start_x + 70, 60), (indicator_start_x + 70 + indicator_size, 60 + indicator_size), (0, 255, 0) if drinking_detected else (0, 0, 255), -1)
    cv2.rectangle(img, (indicator_start_x + 70, 85), (indicator_start_x + 70 + indicator_size, 85 + indicator_size), (0, 255, 0) if seatbelt_detected else (0, 0, 255), -1)

# Analyzing input video
frame_count = 0
try:
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
                seatbelt_detected = False
                for j in boxes:
                    x1, y1, x2, y2, score, class_index = j.numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    img_crop = img[y1:y2, x1:x2]

                    y_pred, score = prediction_func(img_crop)

                    if y_pred == CLASS_NAMES[0] or class_index > 0:
                        log_activity(f'Seatbelt not worn at frame {frame_count}')
                        seatbelt_detected = True
                        current_time = time.time()
                        # Check if 2 hours have passed since the last alert
                        if current_time - last_alert_time >= 7200:  # 7200 seconds = 2 hours
                            # Send alert
                            # send_sms("Attention: Seatbelt not worn detected!")
                            last_alert_time = current_time  # Update last alert time

                    elif y_pred == CLASS_NAMES[1]:
                        log_activity(f'Seatbelt worn detected at frame {frame_count}')
                        draw_color = COLOR_GREEN

                    if score >= THRESHOLD_SCORE:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, f'{y_pred} {str(score)[:4]}', (x1-10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)

                # Detect smoking and drinking
                result = smoking_drinking_model(img, stream=True)
                smoking_detected = False
                drinking_detected = False
                for info in result:
                    boxes = info.boxes
                    for box in boxes:
                        connfidence = box.conf[0]
                        connfidence = math.ceil(connfidence * 100)
                        Class = int(box.cls[0])
                        if connfidence >= THRESHOLD_SCORE:
                            if Class == 0:
                                log_activity(f'{classnames[Class]} detected with {connfidence}% confidence at frame {frame_count}')
                                smoking_detected = True
                            elif Class == 1:
                                log_activity(f'{classnames[Class]} detected with {connfidence}% confidence at frame {frame_count}')
                                drinking_detected = True
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                            cvzone.putTextRect(img, f'{classnames[Class]} {connfidence}%', [x1+8, y1+100], scale=1.5, thickness=2)

                draw_dashboard(img, smoking_detected, drinking_detected, seatbelt_detected)  # Draw the dashboard rectangle

                out.write(img)  # Write frame to video

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.namedWindow('Video feed', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Video feed', output_width, output_height)
                cv2.imshow('Video feed', img)

        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    # Handle keyboard interrupt gracefully
    pass

finally:
    # Release VideoWriter and close windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()

    print("Script run complete. Results saved to:", OUTPUT_FILE)
