import sys
import torch
import cv2
from pathlib import Path
from random import randint

sys.path.insert(0, '../../yolov5')
from utils.plots import Annotator, colors
from utils.general import scale_boxes, LOGGER
cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.11:554')

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
               vid_path,
               vid_writer,
               vid_write,
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
    _, im0 = cap.read()
    annotator = Annotator(im0, line_width=3, example=str(names))

    if len(dets):
        dets[:, :4] = scale_boxes(im.shape[2:], dets[:, :4], im0.shape)

        # Print results
        for c in dets[:, 5].unique():
            n = (dets[:, 5] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        for *xyxy, conf, cls in reversed(dets):
            c = int(cls)
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))

    im0 = annotator.result()
    cv2.imwrite(f"/workspace/videos/ver_nice{randint(0, 1000)}.jpg", im0)
    print(type(im0))
    vid_write.write(im0)
    # # Save results
    # if dataset.mode == 'image':
    #     cv2.imwrite(save_path, im0)
    #     print("cominghere as well")
    # else:
    #     print(vid_path, save_path)
    #     print("comimg here 1")
    #     if vid_path[i] != save_path:
    #         print("Coming here at all?")
    #         vid_path[i] = save_path
    #         if isinstance(vid_writer[i], cv2.VideoWriter):
    #             print('vid release')
    #             vid_writer[i].release()
    #         if vid_cap:
    #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
    #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #         else:
    #             print('else statement')
    #             fps, w, h = 30, im0.shape[1], im0.shape[0]
    #         print("The save Path is: ", save_path)
    #         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
    #         vid_writer[i] = cv2.VideoWriter('/workspace/videos/something_7.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    #     print("coming here  btw")
    #     vid_writer[i].write(im0)
    
    LOGGER.info(f"{s}{'' if len(dets) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape ' % t)






