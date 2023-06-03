import torch 
import sys
from pathlib import Path
import argparse

from model_loader import Model
from dataloader import load_dataset
from tracker import Tracker
from annotation import annotation

sys.path.insert(0, '../../yolov5/')
YOLOV5_ROOT = Path('/workspace/try1/yolov5/')

from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, Profile, check_file, increment_path)

def processing_predictions(preds, tracker: Tracker, **kwargs):
    for i, dets in enumerate(preds):
        tracks = torch.FloatTensor(tracker(dets))
        print(tracks.shape, dets.shape)
        if tracks.shape[0] != dets.shape[0]:
            continue
        dets = torch.cat((dets, tracks[:, 4:]), dim=1)
        print(dets)
        annotation(pred_index=i, dets=dets, tracks=tracks, **kwargs)

def run(
        weights=YOLOV5_ROOT / 'yolov5s.pt',  # model path or triton URL
        source=YOLOV5_ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data= YOLOV5_ROOT / 'data/coco128.yaml',  # dataset.yaml path
        **kwargs,
):
    source = str(source)
    save_img = not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS )
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)

    # Directories 
    save_dir = increment_path(Path('/workspace/try1/output/') / 'exp', exist_ok=False, mkdir=True)
    # Load Model
    model = Model(weights=weights, data_yaml=data, **kwargs)
    # Initializing the tracker 
    tracker = Tracker()
    # Loading dataloaders
    dataset = load_dataset(source=source, is_url=is_url, stride=model.stride, auto=model.pt, **kwargs)
    
    # Run inference
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        preds, dt = model(path, im, save_dir, dt, **kwargs)
        processing_predictions(preds=preds, 
                               tracker=tracker,
                               dt=dt, 
                               seen=seen, 
                               windows=windows, 
                               path=path, 
                               im=im, 
                               im0s=im0s, 
                               vid_cap=vid_cap, 
                               dataset=dataset, 
                               save_dir=save_dir, 
                               s=s, 
                               names=model.names,
                               is_url=is_url,
        )

if __name__ == "__main__":
    run(source='rtsp://admin:admin@192.168.1.11:554', data='/workspace/try1/yolov5/data/coco128.yaml', imgsz=(640, 640), fp16=False, vid_stride=1, conf_thres=0.15, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000, device=torch.device('cpu'))

