import numpy as np
import torch
from trackers.PE_tracker.tracklet import Tracklet
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

        self.space_conds_last = None
        self.short_space = None
    
    def multi_predict(self):
        # The motion prediction of the tracklet begins at the second frame. The track in the second frame only
        # contains one position and does not contain motion information, so HSMP is not used for prediction.
        # We use the current position of the trajectory as the predicted position.
        if self.frame_id < 3:
            for t in self.tracklets_list:
                t.predict_KF()
                t.pos_history.append(t.xyah)
                t.long_history.append(self.normalize(t.xywh))
                t.age += 1
                t.time_since_update += 1
        else:
            pre_model_t = []
            for t in self.tracklets_list:
                t.predict_KF()
                if t.age > 1:  # The newly initialized track does not contain motion information
                    pre_model_t.append(t)
                else:
                    t.pos_history.append(t.xyah)
                    t.long_history.append(self.normalize(t.xywh))
                    t.age += 1
                    t.time_since_update += 1
            self._multi_predict_model(pre_model_t)
                
    def _multi_predict_model(self, t_list):
        if len(t_list) != 0:
            long_h = np.array([t.get_long_conds() for t in t_list])
            short_s = np.array([self.short_space.copy() for i in range(0, len(t_list))])
            long_h = torch.from_numpy(long_h).float().to(self.device)
            short_s = torch.from_numpy(short_s).float().to(self.device)
            with torch.no_grad():
                if self.opt.model_name == "HSMP":
                    output = self.model(long_h, short_s)
                else:
                    output = self.model(long_h)
                output = output.detach().cpu().numpy()  # norm[x,y,w,h]

            for i in range(0, len(t_list)):
                tracklet = t_list[i]
                pre_box_norm = output[i, :].copy()  # norm[x,y,w,h]
                tracklet.long_history.append(pre_box_norm.copy())
                pre_box = self.normxywh2xyah(pre_box_norm)  # norm[x,y,w,h] --> [x,y,a,h]
                tracklet.pos_history.append(pre_box.copy())
                tracklet.age += 1
                tracklet.time_since_update += 1

    
    def update(self, detections):
        self.frame_id += 1

        high_dets = []
        low_dets = []
        for d in detections:
            if d.score >= self.opt.det_high:
                high_dets.append(d)
            elif d.score >= self.opt.det_low:
                low_dets.append(d)

        self.multi_predict()

        confirmed_tracklets = []
        unconfirmed_tracklets = []
        for t in self.tracklets_list:
            if t.is_confirmed():
                confirmed_tracklets.append(t)
            else:
                unconfirmed_tracklets.append(t)

        #------------------------------------------------------First association----------------------------------------------------
        base_matrix = self.base_metric(confirmed_tracklets, high_dets)
        conf_dist = metric.conf_distance(confirmed_tracklets, high_dets, use_kf_pre=True)
        match11, unmatch_T11, unmatch_D11 = matching.linear_assignment_iou_conf(base_matrix, conf_dist, conf_weight=self.opt.conf_weight_1, iou_threshold=self.opt.iou_thrd_1)

        for t_indx, d_indx in match11:
            tracklet = confirmed_tracklets[t_indx]
            det = high_dets[d_indx]
            tracklet.updata(det)

        if not self.opt.P_Ass:
            unmatch_confirmed_tracklets = []
            for t_indx in unmatch_T11:
                tracklet = confirmed_tracklets[t_indx]
                if tracklet.is_tracked():
                    unmatch_confirmed_tracklets.append(tracklet)
                else:
                    tracklet.mark_missed()
            unmatch_high_dets = [high_dets[i] for i in unmatch_D11]
        else:
            unmatch_confirmed_tracklets_KF = [confirmed_tracklets[i] for i in unmatch_T11]
            unmatch_high_dets_KF = [high_dets[i] for i in unmatch_D11]           
            base_matrix_KF = self.base_metric(unmatch_confirmed_tracklets_KF, unmatch_high_dets_KF, use_kf_pre=True)
            conf_dist_KF = metric.conf_distance(unmatch_confirmed_tracklets_KF, unmatch_high_dets_KF, use_kf_pre=True)
            match12, unmatch_T12, unmatch_D12 = matching.linear_assignment_iou_conf(base_matrix_KF, conf_dist_KF, conf_weight=self.opt.conf_weight_1, iou_threshold=self.opt.iou_thrd_1)

            for t_indx, d_indx in match12:
                tracklet = unmatch_confirmed_tracklets_KF[t_indx]
                det = unmatch_high_dets_KF[d_indx]
                tracklet.updata(det, ass_HSMP=False)

            unmatch_confirmed_tracklets = []
            for t_indx in unmatch_T12:
                tracklet = unmatch_confirmed_tracklets_KF[t_indx]
                if tracklet.is_tracked():
                    unmatch_confirmed_tracklets.append(tracklet)
                else:
                    tracklet.mark_missed()
            unmatch_high_dets = [unmatch_high_dets_KF[i] for i in unmatch_D12]

        #------------------------------------------------------The second association----------------------------------------------------
        base_matrix = self.base_metric(unmatch_confirmed_tracklets, low_dets)
        conf_dist = metric.conf_distance(unmatch_confirmed_tracklets, low_dets, use_kf_pre=False)
        match21, unmatch_T21, unmatch_D21 = matching.linear_assignment_iou_conf(base_matrix, conf_dist, conf_weight=self.opt.conf_weight_2, iou_threshold=self.opt.iou_thrd_2)

        for t_indx, d_indx in match21:
            tracklet = unmatch_confirmed_tracklets[t_indx]
            det = low_dets[d_indx]
            tracklet.updata(det, updata_emb=False)

        if not self.opt.P_Ass:
            for t_indx in unmatch_T21:
                tracklet = unmatch_confirmed_tracklets[t_indx]
                tracklet.mark_missed()
        else:
            unmatch_second_tracklets_KF = [unmatch_confirmed_tracklets[i] for i in unmatch_T21]
            unmatch_low_dets_KF = [low_dets[i] for i in unmatch_D21]
            base_matrix_KF = self.base_metric(unmatch_second_tracklets_KF, unmatch_low_dets_KF, use_kf_pre=True)
            conf_dist_KF = metric.conf_distance(unmatch_second_tracklets_KF, unmatch_low_dets_KF, use_kf_pre=False)
            match22, unmatch_T22, unmatch_D22 = matching.linear_assignment_iou_conf(base_matrix_KF, conf_dist_KF, conf_weight=self.opt.conf_weight_2, iou_threshold=self.opt.iou_thrd_2)

            for t_indx, d_indx in match22:
                tracklet = unmatch_second_tracklets_KF[t_indx]
                det = unmatch_low_dets_KF[d_indx]
                tracklet.updata(det, ass_HSMP=False, updata_emb=False)

            for t_indx in unmatch_T22:
                tracklet = unmatch_second_tracklets_KF[t_indx]
                tracklet.mark_missed()

        #------------------------------------------------------The third connection----------------------------------------------------
        base_matrix = self.base_metric(unconfirmed_tracklets, unmatch_high_dets)
        match33, unmatch_T33, unmatch_D33 = matching.linear_assignment_iou(base_matrix, iou_threshold=self.opt.iou_thrd_3)

        for t_indx, d_indx in match33:
            tracklet = unconfirmed_tracklets[t_indx]
            det = unmatch_high_dets[d_indx]
            tracklet.updata(det)

        for t_indx in unmatch_T33:
            tracklet = unconfirmed_tracklets[t_indx]
            tracklet.mark_missed()

        for d_indx in unmatch_D33:
            det = unmatch_high_dets[d_indx]
            if det.score >= self.opt.new_tracklet_conf:
                new_tracklet = Tracklet(self.opt, det, self._next_id, self.frame_id, self.seq_h, self.seq_w)
                self.tracklets_list.append(new_tracklet)
                self.tracklets_dict[self._next_id] = new_tracklet
                self._next_id += 1

        new_tracklets_list = []
        for t in self.tracklets_list:
            if not t.is_deleted():
                new_tracklets_list.append(t)
            else:
                del self.tracklets_dict[t.tracklet_id]
        self.tracklets_list = new_tracklets_list

        self.updata_short_space()


    def updata_short_space(self):
        space_conds_pos = np.array([t.normxywh for t in self.tracklets_list])
        space_conds_id = np.array([[t.tracklet_id] for t in self.tracklets_list])
        space_conds = np.hstack((space_conds_id, space_conds_pos))
        if self.frame_id >= 2:
            self._get_short_space_conds(space_conds)
        if space_conds.shape[0] == 0:
            self.space_conds_last = np.zeros((1, 5))
        else:
            self.space_conds_last = space_conds.copy()

    def _get_short_space_conds(self, space_conds):
        if space_conds.shape[0] == 0:
            self.short_space = np.zeros((1, 8))
        else:
            space_condition = []
            for i in space_conds:
                tid = i[0]
                j = self.space_conds_last[self.space_conds_last[:,0] == tid]
                if j.size != 0:
                    j = j[0]
                    delta = i[1:5] - j[1:5]
                else:
                    delta = np.zeros(4)
                delta_tid = np.hstack((i[1:5], delta))
                space_condition.append(delta_tid)
            space_condition = np.vstack(space_condition)
            self.short_space = space_condition.copy()


    def normalize(self, bbox):
        res = bbox.copy()
        res[0] /= self.seq_w
        res[1] /= self.seq_h
        res[2] /= self.seq_w
        res[3] /= self.seq_h

        return res
    
    def normxywh2xyah(self, bbox):
        res = bbox.copy()
        res[0] *= self.seq_w
        res[1] *= self.seq_h
        res[2] *= self.seq_w
        res[3] *= self.seq_h
        res[2] /= res[3]

        return res








