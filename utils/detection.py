import numpy as np


class Detection(object):
    def __init__(self, tlbr, score, obj_conf, cls_conf, emb=None):
        self._tlbr = np.array(tlbr, dtype=float)
        self._score = float(score)
        self._obj_conf = float(obj_conf)
        self._cls_conf = float(cls_conf)
        if emb is not None:
            self._emb = np.array(emb, dtype=float)
        else:
            self._emb = None

    @property
    def score(self):
        return self._score
    
    @property
    def obj_conf(self):
        return self._obj_conf
    
    @property
    def cls_conf(self):
        return self._cls_conf
    
    @property
    def emb(self):
        if self._emb is not None:
            return self._emb.copy()
        else:
            return None


    @property
    def tlbr(self):
        """
        Retrun det_box [x1,y1,x2,y2]
        """
        return self._tlbr.copy()
    
    @property
    def tlwh(self):
        """
        Retrun det_box [x1,y1,w,h]
        """
        res = self._tlbr.copy()
        res[2] = res[2] - res[0]
        res[3] = res[3] - res[1]

        return res

    @property
    def xyah(self):
        """
        Retrun det_box [cx,cy,a=w/h,h]
        """
        res = self.tlwh
        res[:2] += res[2:] / 2
        res[2] /= res[3]

        return res
    
    @property
    def xywh(self):
        """
        Retrun det_box [cx,cy,w,h]
        """
        res = self.tlwh
        res[:2] += res[2:] / 2
        
        return res
    
    @property
    def xyah_s(self):
        res = self.xyah
        res = np.append(res, self.score)

        return res
    
    @property
    def tlbr_s(self):
        res = self.tlbr
        res = np.append(res, self.score)

        return res