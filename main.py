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


def predict_vehicle_class(vehicle_img, predict_model):
    class_names = {0: "bus", 1: "car", 2: "motorcycle", 3: "truck"}
    vehicle_img = cv2.resize(vehicle_img, (224, 224))
    vehicle_img = np.expand_dims(vehicle_img, axis=0)
    prediction = predict_model.predict(vehicle_img, verbose=0)
    vehicle_class = np.argmax(prediction[0])
    vehicle_class = class_names[vehicle_class]
    return vehicle_class


def main(conf=0.7):
    predict_path = Path("output/best_model.keras")
    demo_video = Path("demo2.mp4")
    output_video = "official_demo3.mov"

    print("Loading model...")
    predict_model = load_model(predict_path, custom_objects={'f1_score': f1_score})
    detect_model = YOLO('yolov8n.pt')
    print('Model loaded successfully.')

    cap = cv2.VideoCapture(str(demo_video))

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    id_classname_dict = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Detect and track vehicles
        results = detect_model.track(frame, conf=conf, persist=True, verbose=False)
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = map(int, box)
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2

            if track_id not in id_classname_dict:
                vehicle_img = frame[y1:y2, x1:x2]
                vehicle_class = predict_vehicle_class(vehicle_img, predict_model)
                id_classname_dict[track_id] = vehicle_class
            else:
                vehicle_class = id_classname_dict[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id} {vehicle_class}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

        # Write the frame to the output video
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('YOLOv8 Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
