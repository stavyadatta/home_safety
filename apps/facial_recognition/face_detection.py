import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '../../utils/')
from dataloader import load_dataset
from tracker import Tracker
from annotation import annotation

sys.path.insert(0, '../../yolov5/')
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, Profile, check_file, increment_path)

sys.path.insert(0, '../../insightface/detection/scrfd/tools/')
from scrfd import SCRFD

class FaceDetection():
    def __init__(self):
        self.annotation_parameters = []

    def update_annotation_params(self, annotation_parameter):
        self.annotation_parameters.append(annotation_parameter)

    def annotation_intrusion(self):
        for param in self.annotation_parameters:
            annotation(**param)

    def draw_detected_boxes(self, im, bboxes: np.ndarray):
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

    @staticmethod
    def crop_region(img, bbox: np.ndarray):
        x1, y1, x2, y2, _ = bbox.astype(np.int32)
        cropped_region = img[y1:y2, x1:x2]
        return cropped_region

    @staticmethod
    def img_run(
        img: np.ndarray,
        weights='/workspace/home_safety/models/scrfd_500m.onnx',
        ):
        ''' 
            Running it on a single image
        '''
        detector = SCRFD(model_file=weights)
        detector.prepare(-1)
        bboxes, _ = detector.detect(img, input_size=(640, 640))
        return bboxes

    def crop_images_and_save(self, bboxes: np.ndarray, img, output_path: Path, img_path: Path):
        save_directory = increment_path(output_path / img_path.stem)
        save_directory.mkdir()

        for bbox_index in range(bboxes.shape[0]):
            bbox = bboxes[bbox_index]
            cropped_region = self.crop_region(img, bbox)
            output_img_index = Path(str(bbox_index) + ".jpg")
            result_output_path = save_directory / output_img_index

            # check if the bbox is empty for some reason
            if 0 in cropped_region.shape: continue
            cv2.imwrite(str(result_output_path.absolute()), cropped_region)

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
        detector = SCRFD(model_file=weights)

        # This is for putting the model on cpu i.e which is -1
        detector.prepare(-1)

        tracker = Tracker()
        dataset = load_dataset(source=source, is_url=is_url, stride=32, imgsz=(640, 640), auto=True, vid_stride=1)
        
        for path, im, im0s, vid_cap, s in dataset:
            bboxes, kpss = detector.detect(im0s, input_size=(640,640))
            self.crop_images_and_save(bboxes, im0s, save_dir, Path(path))
            # im = self.draw_detected_boxes(im0s, bboxes)
            # self.save_img(im, save_dir, path)

if __name__ == "__main__":
    face = FaceDetection()
    face.run(source='../../../try1/sample_video.mp4')
