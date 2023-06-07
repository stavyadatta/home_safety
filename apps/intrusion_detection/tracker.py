import sys
import torch
from collections import defaultdict

sys.path.insert(0, '../../sort/')

from sort import Sort

class Tracker(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.tracker = Sort(max_age, min_hits, iou_threshold)
        self.track_dict = defaultdict(TrackerInstance)

    def __call__(self, dets):
        tracks = self.tracker.update(dets[:,:5])
        self._update_track_dict(tracks)
        return tracks

    def _update_track_dict(self, tracks):
        id_col = tracks[:, -1]
        for id in id_col:
            self.track_dict[id].add(tracks[:, :-1])

class TrackerInstance(object):
    def _centroid(self, det):
        x1 = det[:, 0]
        y1 = det[:, 1]
        x2 = det[:, 2]
        y2 = det[:, 3]

        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2

        centroid = torch.stack((centroid_x, centroid_y), dim=1)
        return centroid

    def add(self, det):
        if not self.dets:
            self.dets = torch.unsqueeze(torch.Tensor(det), dim=0)
            first_centroid = self._centroid(self.dets[0])
            self.centroid_array = torch.unsqueeze(first_centroid, dim=0)
        else:
            self.dets = torch.cat((self.dets, det), dim=0)
            centroid = self._centroid(det)
            self.centroid_array = torch.cat((self.centroid_array, centroid), dim=0)
    
    def is_crossing(self, x1, y1, x2, y2):
        # Checking if the line segment is crossing the dets points
        # The Idea is to calculate the direction of segment from x1, if the direction of the points 
        # with respect to the line segment changes, the point lies on the opposite side of the line segment
        line_vector = torch.Tensor([x2 - x1, y2 - y1])
        point_vector = self.dets - torch.Tensor([x1, y1])
        crossing = torch.cross(line_vector, point_vector)
        return (crossing[:, 0] * crossing[:, 1] < 0).any()
        




            
