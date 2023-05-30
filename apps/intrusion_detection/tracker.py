import sys

sys.path.insert(0, '../../sort/')
sys.path.insert(1, '../../yolov5')

from utils.general import scale_boxes
from sort import Sort

class Tracker(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age, min_hits, iou_threshold)

    def __call__(self, dets):
        return self.tracker.update(dets[:,:5])

            
