import sys
import cv2
import onnx
import pickle
import numpy as np
import onnxruntime as ort
from pathlib import Path

from face_detection import FaceDetection

class DatabaseVectorDevelopment():
    def __init__(self) -> None:
        pass

    def onnx_face_recognition_model_init(self, weights:  str):
        ort_session = ort.InferenceSession(weights, 
                                           providers=['CPUExecutionProvider'])
        return ort_session
    
    @staticmethod
    def preprocess_img(img: np.ndarray):
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def cropped_image(self, img: np.ndarray):
        '''
            Returning cropped images from the imgs
            Assumption - The image is going to contain only a single person
        '''
        bboxes = FaceDetection.img_run(img)
        cropped_img = FaceDetection.crop_region(img, bboxes[0])
        return cropped_img

    def embedding_output(self, cropped_img: np.ndarray, ort_session: ort.InferenceSession):
        img = self.preprocess_img(cropped_img)
        embeddings: np.ndarray = ort_session.run(None, {'data': img})[0]
        return embeddings
    
    @staticmethod
    def embedding_save(embedding: np.ndarray, output_path: Path, img_path: Path):
        embedding_save_name = str(img_path.stem) + ".npy"
        output_path_dir_name = output_path / img_path.parent.stem
        output_path_dir_name.mkdir(exist_ok=True, parents=True)
        full_path = output_path_dir_name / embedding_save_name
        np.save(full_path, embedding)

    def database_imgs2embedding_and_save(self, database_path: Path, output_path: Path, weights: str):
        img_files = list(database_path.glob("*.jpg"))
        ort_session = self.onnx_face_recognition_model_init(weights)
        for img_file in img_files:
            img = cv2.imread(str(img_file.absolute()))
            cropped_img = self.cropped_image(img)
            embedding = self.embedding_output(cropped_img, ort_session)
            self.embedding_save(embedding, output_path, img_file)

if __name__ == "__main__":
    result = DatabaseVectorDevelopment()
    database_path = Path('/workspace/home_safety/database_faces/cp_plus_faces/')
    output_path = Path("/workspace/home_safety/database_embedding")
    weights = '/workspace/home_safety/models/r18_magface_finetune_indian_dataset_IMDB_dynamic_batching_preprocssed.onnx'
    result.database_imgs2embedding_and_save(database_path, output_path, weights)



        
