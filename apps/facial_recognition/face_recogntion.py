import sys
from typing import List
import cv2
import onnx
import pickle
import numpy as np
import onnxruntime as ort
from pathlib import Path

from face_detection import FaceDetection
from database_vector_development import DatabaseVectorDevelopment

sys.path.insert(0, '../../utils/')
from dataloader import load_dataset
from tracker import Tracker
from annotation import annotation
from classes.embedding_list import EmbeddingArray

sys.path.insert(0, '../../yolov5/')
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, Profile, check_file, increment_path)

class FacialRecognition():
    def __init__(self) -> None:
        pass

    @staticmethod
    def img2cropped_imgs(img: np.ndarray):
        bboxes = FaceDetection.img_run(img)
        cropped_imgs = []
        for bbox_index in range(bboxes.shape[0]):
            bbox = bboxes[bbox_index]
            cropped_img = FaceDetection.crop_region(img, bbox)
            cropped_imgs.append(cropped_img)
        return cropped_imgs

    @staticmethod
    def preprocessing_crop_list(cropped_imgs: list):
        return [DatabaseVectorDevelopment.preprocess_img(cropped_img) for cropped_img in cropped_imgs]
    
    @staticmethod
    def prep_embedding_dataset(embedding_dataset_path: Path) -> EmbeddingArray:
        embedding_list = list(embedding_dataset_path.glob("*.npy"))
        embedding_list_array = EmbeddingArray(embedding_list)
        return embedding_list_array

    @staticmethod
    def fr_on_crops(preprocessed_crops: list, face_recognition_ort_session: ort.InferenceSession) -> List[np.ndarray]:
        embedding_list = []
        for preprocessing_crop in preprocessed_crops:
            embedding = face_recognition_ort_session.run(None, {'data': preprocessing_crop})[0]
            embedding_list.append(embedding)
        return embedding_list

    def stream_run(
        self,
        source,
        weights: str,
        save_path: str
        ):
        source = str(source)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS )
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url and is_file:
            source = check_file(source)

        save_dir = increment_path(Path(save_path))
        dataset = load_dataset(source=source, is_url=is_url, stride=32, imgsz=(640, 640), auto=True, vid_stride=1)
        embedding_dataset_path = Path('/workspace/home_safety/database_embedding/cp_plus_faces/')
        embedding_dataset_array = self.prep_embedding_dataset(embedding_dataset_path)

        face_recognition_ort_session = DatabaseVectorDevelopment.onnx_face_recognition_model_init(weights)

        for path, im, im0s, vid_cap, s in dataset:
            cropped_imgs = self.img2cropped_imgs(im0s)
            preprocessed_crops  = self.preprocessing_crop_list(cropped_imgs)
            stream_embedding_list = self.fr_on_crops(preprocessed_crops, face_recognition_ort_session)
            stream_embedding_array = EmbeddingArray(stream_embedding_list)
            minimum_distance_dataset_array = stream_embedding_array.compare_embedding(embedding_dataset_array)
            
