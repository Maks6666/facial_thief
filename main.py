import os
import shutil
from generator_model import model
from generate import generate

from sort import Sort
from ultralytics import YOLO
import numpy as np
import cv2
import torch

class SortTracker:
    def __init__(self, path, device, yolo, photo_amount, remove=False):
        self.path = path
        self.device = device
        self.yolo = yolo
        self.model = self.load_model()
        self.names = self.model.names
        self.sort = Sort(max_age=50, min_hits=8, iou_threshold=0.4)
        self.coef = 50
        self.photo_amount = photo_amount
        self.remove = remove

    def load_model(self):
        model = YOLO(self.yolo)
        model.to(self.device)
        model.fuse()
        return model

    def results(self, frame):
        results = self.model(frame)[0]
        return results

    def get_results(self, results):
        res_arr = []
        if results is not None and len(results) > 0:
            for result in results:
                bboxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_id = result.boxes.cls.cpu().numpy()

                arr = [bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3], scores[0], class_id[0]]
                res_arr.append(arr)

            return np.array(res_arr)
        return np.array(res_arr)

    def draw(self, frame, bboxes, idc, clss):
        for bbox, idx, cls in zip(bboxes, idc, clss):

            name = self.names[int(cls)]
            text = f"{idx}:{name}"

            cv2.rectangle(frame, (int(bbox[0])-self.coef, int(bbox[1])-self.coef), (int(bbox[2])+self.coef, int(bbox[3])+self.coef), (0, 255, 0), 2)
            cv2.putText(frame, text, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def extract(self, bboxes, idc, frame):
        img, idx = None, None
        for bbox, idx in zip(bboxes, idc):
            if os.path.isdir(os.path.join("/Users/maxkucher/opencv/facial_thief/generation", str(idx))):
                continue
            else:
                os.mkdir(os.path.join("/Users/maxkucher/opencv/facial_thief/generation", str(idx)))
                x1, y1, x2, y2 = map(int, bbox)
                img = frame[y1:y2, x1:x2]
                img = cv2.resize(img, (128, 128))
                cv2.imwrite(os.path.join("/Users/maxkucher/opencv/facial_thief/generation", str(idx), f"{idx}.jpg"),
                            img)

        return img, idx

    def generate_images(self, img, idx):
        if img is not None and idx is not None:
            img = torch.tensor(img, dtype=torch.float32).to(self.device)
            img = img.permute(2, 0, 1).unsqueeze(0)
            dir = os.path.join('/Users/maxkucher/opencv/facial_thief/generation', str(idx))
            generate(img, self.photo_amount, dir)

    def __call__(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()

        if self.remove == True:
            path = "/Users/maxkucher/opencv/facial_thief/generation"
            lst = os.listdir(path)
            for obj in lst:
                shutil.rmtree(os.path.join(path, obj))

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            results = self.results(frame)

            results_array = self.get_results(results)

            if len(results_array) == 0:
                results_array = np.empty((0, 5))

            res = self.sort.update(results_array)

            bboxes = res[:, :-1]
            idc = res[:, -1].astype(int)
            clss = results_array[:, -1].astype(int)

            img, idx = self.extract(bboxes, idc, frame)
            self.generate_images(img, idx)


            # frame = frame()
            upd_frame = self.draw(frame, bboxes, idc, clss)
            cv2.imshow('Video', upd_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

path = 1
device = "mps" if torch.backends.mps.is_available() else "cpu"
# you may use any YOLO model here
yolo = "/Users/maxkucher/opencv/facial_thief/yolov11n-face.pt"
tracker = SortTracker(path, device, yolo, 5, remove=True)
tracker()