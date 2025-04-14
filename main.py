import os
import shutil
from generator_model import model
from generate import generate

from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import numpy as np
import cv2
import torch

class DeepTracker:
    def __init__(self, path, device, threshold, photos_amount, remove=True):
        self.path = path
        self.device = device
        self.tracker = DeepSort(max_iou_distance=0.5, max_age=10)
        self.model = self.load_model()
        self.threshold = threshold
        self.names = self.model.names
        self.remove = remove
        self.root_path = "/Users/maxkucher/opencv/facial_thief/generation"
        self.photos_amount = photos_amount

    def load_model(self):
        model = YOLO("/Users/maxkucher/opencv/facial_thief/yolov11n-face.pt")
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame):
        return self.model(frame)[0]

    def get_results(self, results, frame):
        array = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold:
                array.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(score), int(class_id)))

            tracks = self.tracker.update_tracks(raw_detections=array, frame=frame)
            detected_objects = []

            for track in tracks:
                bbox = track.to_ltrb()
                idx = track.track_id
                class_id = track.get_det_class()
                detected_objects.append((bbox, idx, class_id))

            return detected_objects

    def draw(self, detected_objects, frame):
        if detected_objects:
            for bbox, idx, cls in detected_objects:
                name = self.names[int(cls)]
                text = f"{idx}:{name}"
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    def remove_dir(self, root):
        if self.remove == True:
            for dir in os.listdir(root):
                shutil.rmtree(os.path.join(root, dir))



    def get_image(self, frame, detected_objects):
        image, idx = None, None
        if detected_objects:
            root = "/Users/maxkucher/opencv/facial_thief/generation"
            for bbox, idx, _ in detected_objects:
                x1, y1, x2, y2 = map(int, bbox)
                path = os.path.join(root, str(idx))

                if os.path.isdir(path):
                    continue
                else:
                    os.makedirs(path, exist_ok=True)
                    file_path = os.path.join(path, f"{idx}.jpg")

                    image = frame[y1:y2, x1:x2]
                    cv2.imwrite(file_path, image)

            return image, idx
        else:
            return image, idx

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        self.remove_dir(self.root_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(frame)
            detected_objects = self.get_results(results, frame)

            img, idx = self.get_image(frame, detected_objects)
            frame = self.draw(detected_objects, frame)
            path = os.path.join(self.root_path, str(idx))
            if img is not None:
                generate(img, self.photos_amount, path)

            if img is not None:
                print(type(img))

            cv2.imshow('Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

path = 1
device = "mps" if torch.backends.mps.is_available() else "cpu"
threshold = 0.4
tracker = DeepTracker(path, device, threshold, 10)
tracker()



