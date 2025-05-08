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
        # APCU
        self.APCU = opt.APCU
        self.beta = opt.beta
        self.lamda = opt.lamda

        self._use_NSA = opt.use_NSA
        self._up_by_measure_conf = opt.KF_measure_conf
        self.sample_len = opt.sample_len

        self._init_fid = fid  # The initial frame id
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
        self.KF_pre = None

        norm_bbox = self.normalize(detection.xyah)
        self.long_history = deque([norm_bbox.copy() for i in range(0, self.sample_len + 1)], maxlen=self.sample_len + 1)  # norm[cx,cy,w,h]

        self.EMA_alpha = opt.EMA_alpha
        self.features = []
        if detection.emb is not None:
            feature = detection.emb
            feature /= np.linalg.norm(feature)
            self.features.append(feature)


    def predict_KF(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        KF_pre_box = self.mean[:4].copy()  # [cx,cy,a,h]
        self.score_kf_pre = np.clip(self.mean[4], self._det_low, 1.0)
        if self.score_old is not None:
            self.score_Linear_pre = np.clip(2 * self.score - self.score_old, self._det_low, self._det_high)
        else:
            self.score_Linear_pre = np.clip(self.score, self._det_low, self._det_high)
        self.KF_pre = self.xyah2tlbr(KF_pre_box)

    def updata(self, detection, ass_HSMP=True, updata_emb=True):
        if self._up_by_measure_conf:
            score = calculate_iou(self.KF_pre, detection.tlbr)
            self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.xyah_s, score)
        else:
            self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.xyah_s, detection.score)

        # position update
        if ass_HSMP:
            if self.APCU:
                S_d = calculate_iou(self.tlbr, detection.tlbr)
                if S_d >= self.lamda:
                    self.pos_history[-1] = detection.xyah
                    self.long_history[-1] = self.normalize(detection.xywh)
                else:
                    S_p = (1 - (self.time_since_update - 1) / self._max_lost) * self.lamda
                    det_box = detection.xywh
                    pre_box = self.xywh
                    factor = (1 - self.beta) * (S_p / (S_p + S_d))
                    APCU_box = factor * pre_box + (1 - factor) * det_box
                    self.pos_history[-1] = self.xywh2xyah(APCU_box)
                    self.long_history[-1] = self.normalize(APCU_box)
            else:
                self.pos_history[-1] = detection.xyah
                self.long_history[-1] = self.normalize(detection.xywh)
        else:
            if self.APCU:
                S_d = calculate_iou(self.tlbr, detection.tlbr)
                if S_d >= self.lamda:
                    self.pos_history[-1] = detection.xyah
                    self.long_history[-1] = self.normalize(detection.xywh)
                elif S_d < 0.2:
                    last_tracked_box = self.xyah2xywh(self.pos_history[-self.time_since_update - 1])
                    det_box = detection.xywh
                    for i in range(0, min(self.time_since_update + 1, self.sample_len +1)):
                        new_box = ((self.time_since_update - i) / self.time_since_update) * (det_box - last_tracked_box) + last_tracked_box
                        self.long_history[-i - 1] = self.normalize(new_box)
                    self.pos_history[-1] = detection.xyah
                else:
                    S_p = (1 - (self.time_since_update - 1) / self._max_lost) * self.lamda
                    det_box = detection.xywh
                    pre_box = self.xywh
                    factor = (1 - self.beta) * (S_p / (S_p + S_d))
                    APCU_box = factor * pre_box + (1 - factor) * det_box
                    self.pos_history[-1] = self.xywh2xyah(APCU_box)
                    self.long_history[-1] = self.normalize(APCU_box)
            else:
                self.pos_history[-1] = detection.xyah
                self.long_history[-1] = self.normalize(detection.xywh)
        
        # Feature update
        if detection.emb is not None and updata_emb == True:
            smooth_feat = self.EMA_alpha * self.features[-1] + (1 - self.EMA_alpha) * detection.emb
            smooth_feat /= np.linalg.norm(smooth_feat)
            self.features[-1] = smooth_feat

        # Score update
        self.score_old = self.score
        self.score = detection.score

        self.time_since_update = 0
        self._hits += 1
        # State update
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
    
    @property
    def normxywh(self):
        res = self.xywh
        res = self.normalize(res)

        return res


    def get_long_conds(self):
        long_h = np.array(self.long_history)
        long_t1 = long_h[:-1, :].copy()
        long_t2 = long_h[1:, :].copy()
        delta = long_t2 - long_t1
        long_conds = np.hstack((long_t2,delta))

        return long_conds
    

    def normalize(self, bbox):
        res = bbox.copy()
        res[0] /= self._img_w
        res[1] /= self._img_h
        res[2] /= self._img_w
        res[3] /= self._img_h

        return res
    

    def xywh2xyah(self, bbox):
        res = bbox.copy()
        res[2] = res[2] / res[3]
        
        return res
    
    def xyah2xywh(self, bbox):
        res = bbox.copy()
        res[2] = res[2] * res[3]
        
        return res

    def xyah2tlbr(self, bbox):
        res = bbox.copy()
        res[2] *= res[3]
        res[0] -= 0.5 * res[2]
        res[1] -= 0.5 * res[3]
        res[2] += res[0]
        res[3] += res[1]
        
        return res
















