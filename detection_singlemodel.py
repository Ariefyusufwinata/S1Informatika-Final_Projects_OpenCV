import cv2
import time
import numpy as np
import mediapipe as mp
from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path="MobileNetV2_1e-3.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'][1:3]

labels = ["mengantuk", "normal", "distraksi"]

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

            print(f"Prediksi: {label_text}")

            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-time FER", frame)
    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
