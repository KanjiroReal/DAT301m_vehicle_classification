from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from ultralytics import YOLO


def f1_score(y_true, y_pred):
    y_pred = backend.round(y_pred)
    tp = backend.sum(backend.cast(y_true * y_pred, 'float'), axis=0)
    fp = backend.sum(backend.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = backend.sum(backend.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())

    f1 = 2 * p * r / (p + r + backend.epsilon())
    f1 = backend.mean(f1)
    return f1


def predict_vehicle_classes(bboxes, predict_model, frame):
    class_names = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}
    class_names_predicted = []
    for bbox in bboxes:
        bbox = list(map(int, bbox))
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        vehicle_img = frame[y:y + h, x: x + w]
        vehicle_img = cv2.resize(vehicle_img, (224, 224))
        vehicle_img = np.expand_dims(vehicle_img, axis=0)
        prediction = predict_model.predict(vehicle_img, verbose=0)
        vehicle_class = np.argmax(prediction[0])
        vehicle_class = class_names[vehicle_class]
        class_names_predicted.append(vehicle_class)
    return class_names_predicted


def main(conf=0.5):
    predict_path = Path("output/best_model.keras")
    demo_video = Path("demo1.mp4")

    print("Loading model...")
    predict_model = load_model(predict_path, custom_objects={'f1_score': f1_score})
    detect_model = YOLO('yolov8n.pt')
    print('Model loaded successfully.')

    cap = cv2.VideoCapture(str(demo_video))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detect and track vehicles
        results = detect_model.track(frame, conf=conf, persist=True, verbose=False)
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        vehicle_classes = predict_vehicle_classes(boxes, predict_model, frame)

        for box, track_id, vehicle_class in zip(boxes, track_ids, vehicle_classes):
            x, y, w, h = map(int, box)
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id} {vehicle_class}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('YOLOv8 Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
