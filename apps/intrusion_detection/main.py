import os
from pandas.core.dtypes.common import classes 
import torch 
import sys
from pathlib import Path
import argparse

from model_loader import Model
from dataloader import load_dataset

sys.path.insert(0, '../../yolov5/')
YOLOV5_ROOT = Path('/workspace/try1/yolov5/')

from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, Profile, check_file, increment_path)


def run(
        weights=YOLOV5_ROOT / 'yolov5s.pt',  # model path or triton URL
        source=YOLOV5_ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data= YOLOV5_ROOT / 'data/coco128.yaml',  # dataset.yaml path
        **kwargs,
        # imgsz=(640, 640),  # inference size (height, width)
        # conf_thres=0.25,  # confidence threshold
        # iou_thres=0.45,  # NMS IOU threshold
        # max_det=1000,  # maximum detections per image
        # device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # view_img=False,  # show results
        # save_txt=False,  # save results to *.txt
        # save_conf=False,  # save confidences in --save-txt labels
        # save_crop=False,  # save cropped prediction boxes
        # nosave=False,  # do not save images/videos
        # classes=None,  # filter by class: --class 0, or --class 0 2 3
        # agnostic_nms=False,  # class-agnostic NMS
        # augment=False,  # augmented inference
        # visualize=False,  # visualize features
        # update=False,  # update all models
        # project=ROOT / 'runs/detect',  # save results to project/name
        # name='exp',  # save results to project/name
        # exist_ok=False,  # existing project/name ok, do not increment
        # line_thickness=3,  # bounding box thickness (pixels)
        # hide_labels=False,  # hide labels
        # hide_conf=False,  # hide confidences
        # half=True,  # use FP16 half-precision inference
        # dnn=False,  # use OpenCV DNN for ONNX inference
        # vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS )
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)

    # Directories 
    save_dir = increment_path(Path(YOLOV5_ROOT / 'runs/detect') / 'exp', exist_ok=False)
    
    # Load Model
    model = Model(weights=weights, data_yaml=data, **kwargs)
    
    # Loading dataloaders
    dataset = load_dataset(source=source, is_url=is_url, stride=model.stride, auto=model.pt, **kwargs)
    
    # Run inference
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        preds, dt = model(path, im, save_dir, dt, **kwargs)
        print(preds)
        

if __name__ == "__main__":
    run(source='rtsp://admin:admin@192.168.1.11:554', data='/workspace/try1/yolov5/data/coco128.yaml', imgsz=(640, 640), fp16=False, vid_stride=1, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000, device=torch.)

