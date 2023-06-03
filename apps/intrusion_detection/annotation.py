import sys
import torch
import cv2
from pathlib import Path
from random import randint

sys.path.insert(0, '../../yolov5')
from utils.plots import Annotator, colors
from utils.general import scale_boxes, LOGGER, increment_path

def annotation(dets, 
               pred_index,
               tracks,
               dt,
               seen,
               windows,
               path,
               im,
               im0s,
               vid_cap,
               dataset,
               save_dir: Path,
               s,
               names,
               is_url,
):
    seen += 1
    i = pred_index
    if is_url:  # batch_size >= 1
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
    else:
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

    p = Path(p)
    save_path = str(save_dir / p.name) + '.jpg'
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
    s += '%gx%g ' % im.shape[2:]  # print string
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=3, example=str(names), font_size=5)

    if len(dets):
        dets[:, :4] = scale_boxes(im.shape[2:], dets[:, :4], im0.shape)

        # Print results
        for c in dets[:, 5].unique():
            n = (dets[:, 5] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write the results in frame
        for *xyxy, conf, cls, track_id in dets:
            c = int(cls)
            label = f'{names[c]} {conf:.2f}, {track_id}'
            annotator.box_label(xyxy, label, color=colors(c, True))

    # Saving the image in directories
    im0 = annotator.result()
    save_path = increment_path(save_path)
    cv2.imwrite(str(save_path.absolute()), im0)




