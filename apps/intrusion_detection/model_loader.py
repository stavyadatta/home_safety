import os
import torch 
import sys
from pathlib import Path

sys.path.insert(0, '../../yolov5/')

from utils.general import (check_img_size, increment_path, non_max_suppression)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

class Model(object):
    def __init__(self, weights, device='', dnn=False, data_yaml='/workspace/try1/yolov5/data/coco128.yaml', fp16=False, imgsz=(640, 640), **kwargs):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights=weights, device=device, dnn=dnn, data=data_yaml, fp16=fp16)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz=imgsz, s=self.stride)
        
        # Batchsize
        bs = 1
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *imgsz))

    def __call__(self, path, im, save_dir, dt, **kwargs):
        with dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            # print(path)
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) 
            pred = self.model(im, augment=False, visualize=False)

        with dt[2]:
            pred = non_max_suppression(pred, **kwargs)
        return pred, dt



