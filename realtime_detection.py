import cv2
import time
import pygame
import random
import numpy as np
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter

labels = ["mengantuk", "normal", "distraksi"]

model_paths = {
    "1": ("asset/model/MobileNetV2_1e-3.tflite", "MobileNetV2"),
    "2": ("asset/model/MobileNetV3Large_1e-3.tflite", "MobileNetV3Large")
}

def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape'][1:3]
    return interpreter, input_details, output_details, input_shape

def play_alert(label):
    if label == "mengantuk":
        alert_file = f"asset/alert/drowsy_{random.randint(1, 3)}.mp3"
    elif label == "distraksi":
        alert_file = f"asset/alert/distracted_{random.randint(1, 3)}.mp3"
    else:
        return

    try:
        if not pygame.mixer.get_busy():
            pygame.mixer.music.load(alert_file)
            pygame.mixer.music.play()
    except Exception as e:
        print(f"❌ Gagal memutar suara: {e}")

pygame.mixer.init()

model_key = "1"
interpreter, input_details, output_details, input_shape = load_model(model_paths[model_key][0])
active_model_name = model_paths[model_key][1]

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

def preprocess_face(image, bbox):
    x, y, w, h = bbox
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, input_shape)
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    return face

last_prediction = ""
last_alert_time = 5
alert_interval = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            x, y = max(x, 0), max(y, 0)
            width, height = min(width, w - x), min(height, h - y)

            face_img = preprocess_face(frame, (x, y, width, height))

            interpreter.set_tensor(input_details[0]['index'], face_img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = int(np.argmax(output_data))
            label_text = labels[prediction]

            print(f"[{active_model_name}] Prediksi: {label_text}")

            # Cek apakah perlu memutar alert
            current_time = time.time()
            if label_text in ["mengantuk", "distraksi"] and (label_text != last_prediction or current_time - last_alert_time > alert_interval):
                play_alert(label_text)
                last_prediction = label_text
                last_alert_time = current_time

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_text} ({active_model_name})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Real-time FER", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key != -1:
        try:
            key_char = chr(key)
            if key_char in model_paths:
                model_key = key_char
                interpreter, input_details, output_details, input_shape = load_model(model_paths[model_key][0])
                active_model_name = model_paths[model_key][1]
                print(f"✅ Model diganti ke: {active_model_name}")
        except ValueError:
            pass

    time.sleep(0.1)  # untuk 10 FPS

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
