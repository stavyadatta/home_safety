import os 
import torch
import sys

sys.path.insert(0, '../../yolov5/')

from utils.dataloaders import LoadImages, LoadStreams, LoadScreenshots
from utils.general import (LOGGER)

def load_dataset(source, is_url, stride, imgsz, auto, vid_stride):
    bs = 1
    if is_url:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=auto, vid_stride=vid_stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=auto, vid_stride=vid_stride)
    
    return dataset
