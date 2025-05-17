import lap
import numpy as np


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def linear_assignment_iou(iou_matrix, iou_threshold):
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            cost_matrix = -iou_matrix
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            matched_indices = np.array([[y[i], i] for i in x if i >= 0])
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(0, iou_matrix.shape[1]):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)
    unmatched_tracklets = []
    for t in range(0, iou_matrix.shape[0]):
        if t not in matched_indices[:, 0]:
            unmatched_tracklets.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_tracklets.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    u_tracklet = np.array(unmatched_tracklets)
    u_detection = np.array(unmatched_detections)

    return matches, u_tracklet, u_detection


def linear_assignment_iou_conf(iou_matrix, conf_cost, conf_weight, iou_threshold):
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            cost_matrix = conf_weight * conf_cost - iou_matrix
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            matched_indices = np.array([[y[i], i] for i in x if i >= 0])
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(0, iou_matrix.shape[1]):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)
    unmatched_tracklets = []
    for t in range(0, iou_matrix.shape[0]):
        if t not in matched_indices[:, 0]:
            unmatched_tracklets.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_tracklets.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    u_tracklet = np.array(unmatched_tracklets)
    u_detection = np.array(unmatched_detections)

    return matches, u_tracklet, u_detection


def linear_assignment_iou_emb(iou_matrix, emb_matrix, emb_weight, iou_threshold):
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            emb_cost = np.maximum(0.0, 1 - emb_matrix)
            cost_matrix = emb_weight * emb_cost - iou_matrix
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            matched_indices = np.array([[y[i], i] for i in x if i >= 0])
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(0, iou_matrix.shape[1]):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)
    unmatched_tracklets = []
    for t in range(0, iou_matrix.shape[0]):
        if t not in matched_indices[:, 0]:
            unmatched_tracklets.append(t)

    # filter out matched with low IOU
    # iou_matrix_thre = iou_matrix - conf_cost
    iou_matrix_thre = iou_matrix
    matches = []
    for m in matched_indices:
        if iou_matrix_thre[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_tracklets.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    u_tracklet = np.array(unmatched_tracklets)
    u_detection = np.array(unmatched_detections)

    return matches, u_tracklet, u_detection


def linear_assignment_iou_conf_emb(iou_matrix, conf_cost, emb_matrix, conf_weight, emb_weight, iou_threshold):
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            emb_cost = np.maximum(0.0, 1 - emb_matrix)
            cost_matrix = conf_weight * conf_cost - iou_matrix + emb_weight * emb_cost
            ## If you test on the MOT17 dataset, you will get better results using the following two lines.
            # emb_cost = emb_matrix * compute_aw_new_metric(emb_matrix)
            # cost_matrix = conf_weight * conf_cost - iou_matrix - emb_weight * emb_cost
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            matched_indices = np.array([[y[i], i] for i in x if i >= 0])
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(0, iou_matrix.shape[1]):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)
    unmatched_tracklets = []
    for t in range(0, iou_matrix.shape[0]):
        if t not in matched_indices[:, 0]:
            unmatched_tracklets.append(t)

    # filter out matched with low IOU
    # iou_matrix_thre = iou_matrix - conf_cost
    iou_matrix_thre = iou_matrix
    matches = []
    for m in matched_indices:
        if iou_matrix_thre[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[1])
            unmatched_tracklets.append(m[0])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    u_tracklet = np.array(unmatched_tracklets)
    u_detection = np.array(unmatched_detections)

    return matches, u_tracklet, u_detection


def compute_aw_new_metric(emb_cost, w_association_emb=2.2, max_diff=1.7):
    w_emb = np.full_like(emb_cost, w_association_emb)
    w_emb_bonus = np.full_like(emb_cost, 0)

    # Needs two columns at least to make sense to boost
    if emb_cost.shape[1] >= 2:
        # Across all rows
        for idx in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[idx])
            # Row weight is difference between top / second top
            row_weight = min(emb_cost[idx, inds[0]] - emb_cost[idx, inds[1]], max_diff)
            # Add to row
            w_emb_bonus[idx] += row_weight / 2

    if emb_cost.shape[0] >= 2:
        for idj in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, idj])
            col_weight = min(emb_cost[inds[0], idj] - emb_cost[inds[1], idj], max_diff)
            w_emb_bonus[:, idj] += col_weight / 2

    return w_emb + w_emb_bonus


















