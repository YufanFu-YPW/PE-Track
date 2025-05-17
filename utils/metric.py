import numpy as np


def iou_batch(tracklets_list, dets_list, use_kf_pre=False):
    o = np.zeros((len(tracklets_list), len(dets_list)), dtype=np.float)
    if o.size == 0:
        return o
    
    bboxes2 = np.array([d.tlbr for d in dets_list])
    if use_kf_pre:
        bboxes1 = np.array([t.KF_pre for t in tracklets_list])
    else:
        bboxes1 = np.array([t.tlbr for t in tracklets_list])
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)                                              
    return o


def hmiou(tracklets_list, dets_list, use_kf_pre=False):
    """
    Height_Modulated_IoU
    """
    o = np.zeros((len(tracklets_list), len(dets_list)), dtype=np.float)
    if o.size == 0:
        return o
    
    bboxes2 = np.array([d.tlbr for d in dets_list])
    if use_kf_pre:
        bboxes1 = np.array([t.KF_pre for t in tracklets_list])
    else:
        bboxes1 = np.array([t.tlbr for t in tracklets_list])
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o = (yy12 - yy11) / (yy22 - yy21)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o *= wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    return o


def diou_batch(tracklets_list, dets_list, use_kf_pre=False):
    o = np.zeros((len(tracklets_list), len(dets_list)), dtype=np.float)
    if o.size == 0:
        return o
    
    bboxes2 = np.array([d.tlbr for d in dets_list])
    if use_kf_pre:
        bboxes1 = np.array([t.KF_pre for t in tracklets_list])
    else:
        bboxes1 = np.array([t.tlbr for t in tracklets_list])
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0 # resize from (-1,1) to (0,1)


def ciou_batch(tracklets_list, dets_list, use_kf_pre=False):
    o = np.zeros((len(tracklets_list), len(dets_list)), dtype=np.float)
    if o.size == 0:
        return o
    
    bboxes2 = np.array([d.tlbr for d in dets_list])
    if use_kf_pre:
        bboxes1 = np.array([t.KF_pre for t in tracklets_list])
    else:
        bboxes1 = np.array([t.tlbr for t in tracklets_list])
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh) 

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    h2 = h2 + 1.
    h1 = h1 + 1.
    arctan = np.arctan(w2/h2) - np.arctan(w1/h1)
    v = (4 / (np.pi ** 2)) * (arctan ** 2)
    S = 1 - iou 
    alpha = v / (S+v)
    ciou = iou - inner_diag / outer_diag - alpha * v
    
    return (ciou + 1) / 2.0 # resize from (-1,1) to (0,1)


def homiou(tracklets_list, dets_list, use_kf_pre=False):
    """
    Hybrid offsets Modulated IOU
    [tlbr]
    """
    o = np.zeros((len(tracklets_list), len(dets_list)), dtype=np.float)
    if o.size == 0:
        return o
    
    bboxes2 = np.array([d.tlbr for d in dets_list])
    if use_kf_pre:
        bboxes1 = np.array([t.KF_pre for t in tracklets_list])
    else:
        bboxes1 = np.array([t.tlbr for t in tracklets_list])
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    # hybrid offsets IoU (HOIoU)
    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    h_union = yy22 - yy21
    delta_c = abs(bboxes1[..., 1] + bboxes1[..., 3] - bboxes2[..., 1] - bboxes2[..., 3]) * 0.5
    delta_d = abs(bboxes1[..., 3] - bboxes2[..., 3])
    hoiou = 1 - (delta_c + delta_d) / h_union
    # iou
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou = wh / ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
                + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)
    # hybrid offsets modulated IoU (HOMIoU)
    return hoiou * iou


def conf_distance(tracklets_list, dets_list, use_kf_pre=True):
    if use_kf_pre:
        t_confs = np.array([t.score_kf_pre for t in tracklets_list])
    else:
        t_confs = np.array([t.score_Linear_pre for t in tracklets_list])
    d_confs = np.array([d.score for d in dets_list])
    t_confs = np.expand_dims(t_confs, 1)
    d_confs = np.expand_dims(d_confs, 0)

    conf_cost = abs(t_confs - d_confs)

    return conf_cost


def emb_matrix(tracklets_list, dets_list):
    t_embs = np.array([t.features[-1] for t in tracklets_list])
    d_embs = np.array([d.emb for d in dets_list])
    matrix = None if (t_embs.shape[0] == 0 or d_embs.shape[0] == 0) else t_embs @ d_embs.T

    return matrix
