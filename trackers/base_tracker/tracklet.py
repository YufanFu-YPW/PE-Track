import numpy as np
# from utils.kalman_filter import KalmanFilter
from utils.kalman_filter_score import KalmanFilterScore
from collections import deque
from utils.tools import calculate_iou


class TrackletState:
    Newtrack = 1
    Confirmed = 2
    Deleted = 3
    Tracked = 4
    Lost = 5


class Tracklet:
    def __init__(self, opt, detection, tracklet_id, fid, img_h, img_w):
        self.tracklet_id = tracklet_id
        self._img_h = img_h
        self._img_w = img_w
        self._det_low = opt.det_low
        self._det_high = opt.det_high
        

        self._use_NSA = opt.use_NSA

        self._init_fid = fid  # 初始的帧id
        self._max_lost = opt.max_lost
        self.age = 1
        self._hits = 1
        self._n_init = opt.n_init
        self.time_since_update = 0

        self.track_state = TrackletState.Tracked
        if fid == 1:
            self.state = TrackletState.Confirmed
        else:
            self.state = TrackletState.Newtrack

        self.score = detection.score
        self.score_old = None
        self.score_kf_pre = None
        self.score_Linear_pre = None

        self.pos_history = []
        self.pos_history.append(detection.xyah)  

        self.kf = KalmanFilterScore(self._use_NSA)
        self.mean, self.covariance = self.kf.initiate(detection.xyah_s)

        self.EMA_alpha = opt.EMA_alpha
        self.features = []
        if detection.emb is not None:
            feature = detection.emb
            feature /= np.linalg.norm(feature)
            self.features.append(feature)


    def predict_KF(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.pos_history.append(self.mean[:4].copy())  # [cx,cy,a,h]
        self.age += 1
        self.time_since_update += 1

        self.score_kf_pre = np.clip(self.mean[4], self._det_low, 1.0)
        if self.score_old is not None:
            self.score_Linear_pre = np.clip(2 * self.score - self.score_old, self._det_low, self._det_high)
        else:
            self.score_Linear_pre = np.clip(self.score, self._det_low, self._det_high)

    def updata(self, detection, updata_emb=True):
        # 位置更新
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.xyah_s, detection.score)
        self.pos_history[-1] = self.mean[:4]
        # 特征更新
        if detection.emb is not None and updata_emb == True:
            smooth_feat = self.EMA_alpha * self.features[-1] + (1 - self.EMA_alpha) * detection.emb
            smooth_feat /= np.linalg.norm(smooth_feat)
            self.features[-1] = smooth_feat

        # 分数更新
        self.score_old = self.score
        self.score = detection.score

        self.time_since_update = 0
        self._hits += 1
        # 状态更新
        self.track_state = TrackletState.Tracked
        if self.state == TrackletState.Newtrack and self._hits >= self._n_init:
            self.state = TrackletState.Confirmed


    def mark_missed(self):
        self.score_old = None
        self.track_state = TrackletState.Lost
        if self.state == TrackletState.Newtrack:
            self.state = TrackletState.Deleted
        elif self.time_since_update > self._max_lost:
            self.state = TrackletState.Deleted


    def is_newtrack(self):
        return self.state == TrackletState.Newtrack

    def is_confirmed(self):
        return self.state == TrackletState.Confirmed

    def is_deleted(self):
        return self.state == TrackletState.Deleted
    
    def is_lost(self):
        return self.track_state == TrackletState.Lost
    
    def is_tracked(self):
        return self.track_state == TrackletState.Tracked 
        
    @property
    def xyah(self):
        return self.pos_history[-1].copy()
    
    @property
    def xywh(self):
        res = self.pos_history[-1].copy()
        res[2] = res[2] * res[3]

        return res

    @property
    def tlwh(self):
        res = self.xywh
        res[0] -= 0.5 * res[2]
        res[1] -= 0.5 * res[3]

        return res
    
    @property
    def tlbr(self):
        res = self.tlwh
        res[2] += res[0]
        res[3] += res[1]

        return res
    

    def xywh2xyah(self, bbox):
        res = bbox.copy()
        res[2] = res[2] / res[3]
        
        return res

    def xyah2tlbr(self, bbox):
        res = bbox.copy()
        res[2] *= res[3]
        res[0] -= 0.5 * res[2]
        res[1] -= 0.5 * res[3]
        res[2] += res[0]
        res[3] += res[1]
        
        return res
















