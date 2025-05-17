import numpy as np
import torch
from trackers.base_tracker.tracklet import Tracklet
from utils import matching
from utils import metric


BASE_METRIC = {
    "iou": metric.iou_batch, 
    "hmiou": metric.hmiou, 
    "HOMIoU": metric.homiou, 
    "diou": metric.diou_batch, 
    "ciou": metric.ciou_batch
}


class Tracker:
    def __init__(self, opt, seq_h, seq_w, motion_model=None, device='cuda:0'):
        self.opt = opt
        self.seq_h = seq_h
        self.seq_w = seq_w
        self.model = motion_model
        self.device = device

        self.base_metric = BASE_METRIC[opt.iou_type]

        self.tracklets_list = []
        self.tracklets_dict = {}
        self._next_id = 1
        self.frame_id = 0
    
    def multi_predict(self):
        for t in self.tracklets_list:
            t.predict_KF()

    def update(self, detections):
        self.frame_id += 1
        # 将检测分为高分和低分
        high_dets = []
        low_dets = []
        for d in detections:
            if d.score >= self.opt.det_high:
                high_dets.append(d)
            elif d.score >= self.opt.det_low:
                low_dets.append(d)

        # 位置预测
        self.multi_predict()

        # 将track分为confirmed和unconfirmed
        confirmed_tracklets = []
        unconfirmed_tracklets = []
        for t in self.tracklets_list:
            if t.is_confirmed():
                confirmed_tracklets.append(t)
            else:
                unconfirmed_tracklets.append(t)

        #------------------------------------------------------第一次关联----------------------------------------------------
        base_matrix = self.base_metric(confirmed_tracklets, high_dets)
        conf_dist = metric.conf_distance(confirmed_tracklets, high_dets, use_kf_pre=True)
        match11, unmatch_T11, unmatch_D11 = matching.linear_assignment_iou_conf(base_matrix, conf_dist, conf_weight=self.opt.conf_weight_1, iou_threshold=self.opt.iou_thrd_1)
        # 更新
        for t_indx, d_indx in match11:
            tracklet = confirmed_tracklets[t_indx]
            det = high_dets[d_indx]
            tracklet.updata(det)
        # 处理未匹配轨迹和检测
        unmatch_confirmed_tracklets = []
        for t_indx in unmatch_T11:
            tracklet = confirmed_tracklets[t_indx]
            if tracklet.is_tracked():
                unmatch_confirmed_tracklets.append(tracklet)
            else:
                tracklet.mark_missed()
        unmatch_high_dets = [high_dets[i] for i in unmatch_D11]

        #------------------------------------------------------第二次关联----------------------------------------------------
        base_matrix = self.base_metric(unmatch_confirmed_tracklets, low_dets)
        conf_dist = metric.conf_distance(unmatch_confirmed_tracklets, low_dets, use_kf_pre=False)
        match21, unmatch_T21, unmatch_D21 = matching.linear_assignment_iou_conf(base_matrix, conf_dist, conf_weight=self.opt.conf_weight_2, iou_threshold=self.opt.iou_thrd_2)
        # 更新
        for t_indx, d_indx in match21:
            tracklet = unmatch_confirmed_tracklets[t_indx]
            det = low_dets[d_indx]
            tracklet.updata(det, updata_emb=False)
        # 处理未匹配轨迹和检测
        for t_indx in unmatch_T21:
            tracklet = unmatch_confirmed_tracklets[t_indx]
            tracklet.mark_missed()

        #------------------------------------------------------第三次关联----------------------------------------------------
        base_matrix = self.base_metric(unconfirmed_tracklets, unmatch_high_dets)
        match33, unmatch_T33, unmatch_D33 = matching.linear_assignment_iou(base_matrix, iou_threshold=self.opt.iou_thrd_3)
        # 匹配成功的进行更新
        for t_indx, d_indx in match33:
            tracklet = unconfirmed_tracklets[t_indx]
            det = unmatch_high_dets[d_indx]
            tracklet.updata(det)

        for t_indx in unmatch_T33:
            tracklet = unconfirmed_tracklets[t_indx]
            tracklet.mark_missed()

        # 初始化新id
        for d_indx in unmatch_D33:
            det = unmatch_high_dets[d_indx]
            if det.score >= self.opt.new_tracklet_conf:
                new_tracklet = Tracklet(self.opt, det, self._next_id, self.frame_id, self.seq_h, self.seq_w)
                self.tracklets_list.append(new_tracklet)
                self.tracklets_dict[self._next_id] = new_tracklet
                self._next_id += 1

        ## 更新
        new_tracklets_list = []
        for t in self.tracklets_list:
            if not t.is_deleted():
                new_tracklets_list.append(t)
            else:
                del self.tracklets_dict[t.tracklet_id]
        self.tracklets_list = new_tracklets_list









