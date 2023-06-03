import sys

sys.path.insert(0, '../../yolov5/')

from utils.dataloaders import LoadImages, LoadStreams, LoadScreenshots

def load_dataset(source, is_url, stride, imgsz, auto, vid_stride, **kwargs):
    bs = 1
    if is_url:
        print("Going to the LoadStreams menu")
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=auto, vid_stride=vid_stride)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=auto, vid_stride=vid_stride)
    
    return dataset, bs
