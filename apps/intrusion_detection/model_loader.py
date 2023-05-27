import os
import torch 
import sys
from pathlib import Path

sys.path.insert(0, '../../yolov5/')

from utils.general import (LOGGER, Profile, check_img_size, check_requirements, non_max_suppression, cv2)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

class Model(object):
    def __init__(self, weights, device='', dnn=False, data_yaml='/workspace/try1/yolov5/data/coco128.yaml', fp16=False, imgsz=(640, 640)):
        device = select_device(device)
        model = DetectMultiBackend(weights=weights, device=device, dnn=dnn, data=data_yaml, fp16=fp16)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz=imgsz

