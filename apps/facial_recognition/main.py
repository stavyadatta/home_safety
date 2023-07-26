from numpy.lib import source
import torch 
import sys
import numpy as np
import argparse
import cv2
from pathlib import Path
from typing import List

sys.path.insert(0, '../../utils/')
from model_loader import Model
from dataloader import load_dataset
from tracker import Tracker
from annotation import annotation

sys.path.insert(0, '../../yolov5/')
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, Profile, check_file, increment_path)

sys.path.insert(0, '../../insightface/detection/scrfd/tools/')
from scrfd import SCRFD

class FacialRecognition():
    def __init__(self):
        self.annotation_parameters = []

    def update_annotation_params(self, annotation_parameter):
        self.annotation_parameters.append(annotation_parameter)

    def annotation_intrusion(self):
        for param in self.annotation_parameters:
            annotation(**param)

    def draw_detected_boxes(self, im, bboxes: List[np.ndarray]):
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1, y1, x2, y2, score = bbox.astype(np.int32)
            cv2.rectangle(im, (x1, y1), (x2, y2), (255,0,0))
        return im

    def save_img(self, im, save_path: Path, img_path: Path):
        save_path = str(save_path / Path(img_path).name) + '.jpg'
        save_path = Path(save_path)
        save_path = increment_path(save_path)
        cv2.imwrite(str(save_path.absolute()), im)
        
    def run(
        self,
        source,
        weights='/workspace/home_safety/models/scrfd_500m.onnx',
        ):
        source=str(source)
        save_img = not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS )
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url and is_file:
            source = check_file(source)

        save_dir = increment_path(Path('/workspace/home_safety/output') / 'exp', exist_ok=False, mkdir=True)
        # Load Model
        detector = SCRFD(model_file=weights)
        # This is for putting the model on cpu i.e which is -1
        detector.prepare(-1)
        # Initializing the tracker 
        tracker = Tracker()
        # Loading dataloaders
        dataset = load_dataset(source=source, is_url=is_url, stride=32, imgsz=(640, 640), auto=True, vid_stride=1)
        
        for path, im, im0s, vid_cap, s in dataset:
            print(im0s.shape)
            cv2.imwrite("/workspace/amazing.jpg", im0s)
            bboxes, kpss = detector.detect(im0s, input_size=(640,640))
            im = self.draw_detected_boxes(im0s, bboxes)
            self.save_img(im, save_dir, path)

if __name__ == "__main__":
    face = FacialRecognition()
    face.run(source='../../../try1/sample_video.mp4')
