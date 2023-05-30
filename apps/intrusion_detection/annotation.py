import sys
import torch
from pathlib import Path

sys.path.insert(0, '../../yolov5')
from utils.plots import Annotator
from utils.general import scale_boxes, LOGGER

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
    save_path = str(save_dir / p.name)
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
    s += '%gx%g ' % im.shape[2:]  # print string
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    imc = im0 # for save_crop
    annotator = Annotator(im0, line_width=3, example=str(names))

    if len(dets):
        dets[:, :4] = scale_boxes(im.shape[2:], dets[:, :4], im0.shape)

        # Print results
        for c in dets[:, 5].unique():
            n = (dets[:, 5] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        LOGGER.info(f"{s}{'' if len(dets) else '(no detections), '}")


