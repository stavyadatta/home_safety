import torch 
import sys
from pathlib import Path
import argparse
import torch._dynamo.config

from model_loader import Model
from dataloader import load_dataset
from tracker import Tracker
from annotation import annotation

sys.path.insert(0, '../../yolov5/')
YOLOV5_ROOT = Path('/workspace/try1/yolov5/')
torch._dynamo.config.suppress_errors = True

from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER, Profile, check_file, increment_path)

class IntrusionDetection():

    def __init__(self):
        self.annotation_parameters = []

    def update_annotation_params(self, annotation_parameter):
        self.annotation_parameters.append(annotation_parameter)

    def annotation_intrusion(self):
        for param in self.annotation_parameters:
            annotation(**param)

    def processing_predictions(self, preds, tracker: Tracker, **kwargs):
        for i, dets in enumerate(preds):
            tracks = torch.FloatTensor(tracker(dets))
            # Checking if the tracks and dets are of same shape to have the tracking consistent
            # TODO - Need to change this logic at later stage
            if tracks.shape[0] != dets.shape[0]:
                continue
            dets = torch.cat((dets, tracks[:, 4:]), dim=1)
            annotation_parameter = {'pred_index':i, 'dets':dets, 'tracks':tracks}
            annotation_parameter.update(kwargs)
            self.update_annotation_params(annotation_parameter)
            self.tracker = Tracker()

    def run(
            self,
            weights= '/workspace/try1/crowdhuman_yolov5m_openvino_model/',
            source=YOLOV5_ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
            data= YOLOV5_ROOT / 'data/person.yaml',  # dathhhhset.yaml path
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
        # Loading dataloaders
        dataset = load_dataset(source=source, is_url=is_url, stride=model.stride, auto=model.pt, **kwargs)
        
        # Run inference
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            preds, dt = model(path, im, save_dir, dt, **kwargs)
            LOGGER.info(f"The timing is {dt[1].dt * 1E3:.1f}ms")
            self.processing_predictions(preds=preds, 
                                tracker=self.tracker,
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
    intrusion_detection = IntrusionDetection()
    try:
        intrusion_detection.run(source='rtsp://admin:admin@192.168.1.11:554', data=Path('/workspace/try1/yolov5/data/person.yaml'), imgsz=(640, 640), fp16=False, vid_stride=1, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000, device=torch.device('cpu'))
    except KeyboardInterrupt:
        intrusion_detection.annotation_intrusion()


