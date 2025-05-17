import time
import colorsys
import cv2
import random
import numpy as np


class Print:
    def __init__(self, log_path):
        self._log_path = log_path

    def log(self, string, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            string = f'[{localtime}] {string}'
        print(string)
        with open(self._log_path, 'a') as f:
            print(string, file=f)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


# Generate Color
def generate_colors(N=500):
    colors = []
    for i in range(N):
        hue = i / N
        rgb_float = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        rgb = tuple(int(c * 255) for c in rgb_float)
        colors.append(rgb)
    random.shuffle(colors)
    return colors


def draw_bbox_id_general(img, bbox, id, color):
    bbox = np.round(bbox).astype(np.int32)

    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, lineType=cv2.LINE_AA)
    img = cv2.putText(img, f'ID={id}', (bbox[0], bbox[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, lineType=cv2.LINE_AA)

    return img


def draw_dashed_bbox_id(img, bbox, id, color):
    bbox = np.round(bbox).astype(np.int32)

    dash_pattern = (5, 15)     # Dashed line mode (line length, spacing)

    for i in range(bbox[0], bbox[2], dash_pattern[1]):
        img = cv2.line(img, (i, bbox[1]), (i + dash_pattern[0], bbox[1]), color, 2, lineType=cv2.LINE_AA)

    for i in range(bbox[1], bbox[3], dash_pattern[1]):
        img = cv2.line(img, (bbox[0], i), (bbox[0], i + dash_pattern[0]), color, 2, lineType=cv2.LINE_AA)

    for i in range(bbox[0], bbox[2], dash_pattern[1]):
        img = cv2.line(img, (i, bbox[3]), (i + dash_pattern[0], bbox[3]), color, 2, lineType=cv2.LINE_AA)

    for i in range(bbox[1], bbox[3], dash_pattern[1]):
        img = cv2.line(img, (bbox[2], i), (bbox[2], i + dash_pattern[0]), color, 2, lineType=cv2.LINE_AA)
    
    img = cv2.putText(img, f'ID={id}', (bbox[0], bbox[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, lineType=cv2.LINE_AA)

    return img


def draw_bbox_id(img, bbox, id, color):
    margin = 3
    bbox = np.round(bbox).astype(np.int32)
    img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, lineType=cv2.LINE_AA)

    text = f'{id}'
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    # (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
    h = (text_height + 2 * margin)
    bg_pt1 = (bbox[0], bbox[1])
    bg_pt2 = (bbox[0] + text_width + 2 * margin, bbox[1] + h)
    img = cv2.rectangle(img, bg_pt1, bg_pt2, color, -1, lineType=cv2.LINE_AA)

    img = cv2.putText(img, text, (bg_pt1[0] + margin, bg_pt2[1] - margin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, lineType=cv2.LINE_AA)
    # img = cv2.putText(img, text, (bg_pt1[0] + margin, bg_pt2[1] - margin), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, lineType=cv2.LINE_AA)

    return img







