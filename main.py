import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import torch
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from utils.det_emb import DetEmb
from utils.detection import Detection
from utils.tools import Print, generate_colors, draw_bbox_id, draw_dashed_bbox_id
from data_process.MOT17.split_label import MOT17_test_seqs, MOT17_train_seqs, MOT17_train_half_len


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--device', default=0)
    parser.add_argument('--weight', default=None)
    parser.add_argument('--name', default=None)
    return parser.parse_args()


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def write_results(save_path, results):
    f = open(save_path, 'w')
    for row in results:
        print(f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},1,-1,-1,-1', file=f)


def load_motion_model(opt, device):
        if opt.model_name == "HSMP":
            from models.HSMP.history_space_model import HSModel
            motion_model = HSModel(opt.d_model, opt.v_size, opt.block_layers, opt.d_state, opt.d_conv, opt.expand, opt.fusion_expand, 
                                   opt.mamba_layers, opt.pre_mamba_layers, opt.bi_mamba, opt.heads, opt.norm_epsilon, opt.rms_norm, opt.dropout)
            model_weight = torch.load(opt.weight_path, map_location = "cpu")
            motion_model.load_state_dict(model_weight['ddpm'])
        elif opt.model_name == "TrackSSM":
            from models.other_models.TrackSSM.track_ssm import TrackSSM
            motion_model = TrackSSM()
            model_weight = torch.load("models/other_models/weights/trackssm/Dance_epoch120.pt", map_location = "cpu")
            motion_model.load_state_dict({k.replace('module.', ''): v for k, v in model_weight['ddpm'].items()})
        elif opt.model_name == "DiffMOT":
            from models.other_models.DiffMOT.diffmot import DiffMOT
            motion_model = DiffMOT()
            model_weight = torch.load("models/other_models/weights/diffmot/DanceTrack_epoch800.pt", map_location = "cpu")
            motion_model.load_state_dict({k.replace('module.', ''): v for k, v in model_weight['ddpm'].items()})
        else:
            return None
        motion_model.eval()
        motion_model.to(device)

        return motion_model


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.device}")
    print(f"use device: cuda {args.device}")

    # Load hyperparameters
    with open(args.config, 'rb') as f:
        opt = yaml.safe_load(f)
    opt = EasyDict(opt)
    if args.weight is not None:
        opt.weight_path = args.weight
    if args.name is not None:
        opt.tracker_name = args.name

    # load seqs
    if opt.datasets == "mot17":
        if opt.mod == "val":
            seqs_list = MOT17_train_seqs
        else:
            seqs_list = MOT17_test_seqs
    else:
        seqs_list = [s for s in os.listdir(opt.data_root)]
    seqs_list.sort()

    # load det_model 和 emb_model
    det_emb_model = DetEmb(opt.load_emb_model)
    if opt.use_cache:
        det_emb_model.load_cache(opt.cache_path)
    else:
        if opt.mod == "val":
            det_emb_model.load_models(opt.det_ckpt, opt.det_exp_path, opt.datasets, emb_test=False, device=device)
        else:
            det_emb_model.load_models(opt.det_ckpt, opt.det_exp_path, opt.datasets, emb_test=True, device=device)

    # Loading the motion model
    motion_model = load_motion_model(opt, device)

    # Load Tracker
    if opt.tracker == "PE-Track":
        from trackers.PE_tracker.tracker import Tracker
    elif opt.tracker == "PE-Track-Reid":
        from trackers.PE_reid_tracker.tracker import Tracker
    else:
        from trackers.base_tracker.tracker import Tracker
        # from trackers.PE_KF_tracker.tracker import Tracker

    tracker_save_dir = os.path.join(opt.save_root, opt.tracker_name)
    results_save_dir = os.path.join(tracker_save_dir, "tracker")
    mkdir(tracker_save_dir)
    mkdir(results_save_dir)
    log_path = os.path.join(tracker_save_dir, "log.txt")
    Plog = Print(log_path)
    # Print arguments
    Plog.log('config:', False)
    for k, v in vars(opt).items():
        Plog.log(f'{k} = {v}', False)

    if opt.visualise:
        id_to_color = generate_colors()

    for seq in seqs_list:
        seq_info = open(os.path.join(opt.data_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        if opt.visualise:
            visual_seq_root = os.path.join(opt.visual_root, "output", seq)
            # visual_seq_pre_root = os.path.join(opt.visual_root, "predict", seq)
            mkdir(visual_seq_root)
            # mkdir(visual_seq_pre_root)

        # print(seq)
        imgs_dir = os.path.join(opt.data_root, seq, 'img1')
        imgs_list = [i for i in os.listdir(imgs_dir)]
        imgs_list.sort()
        # 初始化 Tracker
        tracker = Tracker(opt, seq_height, seq_width, motion_model, device)

        begin_fid = 0
        if opt.datasets == "mot17" and opt.mod == "val":
            begin_fid = MOT17_train_half_len[seq]

        results = []
        for fid in tqdm(range(begin_fid, len(imgs_list)), ncols=100, desc=f"processing {seq}"):
            tag = f"{seq}:{fid + 1}"

            if opt.visualise:
                vis_img_path = os.path.join(imgs_dir, imgs_list[fid])
                vis_img = cv2.imread(vis_img_path)

            # Get dets and reid features
            if opt.use_cache:
                dets, det_embs = det_emb_model.get_data_from_cache(tag)
            else:
                img_path = os.path.join(imgs_dir, imgs_list[fid])
                img = cv2.imread(img_path)
                dets, det_embs = det_emb_model.get_data_from_models(img, (opt.det_input_h, opt.det_input_w), opt.det_conf_thrd, opt.det_nms_thrd, tag)

            # Initialize the detection class
            detections_list = []
            for i in range(0, dets.shape[0]):
                det_box = dets[i, 0:4].copy()  # tlbr:[x1,y1,x2,y2]
                det_score = dets[i, 4]
                det_obj_conf = dets[i, 5]
                det_cls_conf = dets[i, 6]
                if det_embs is not None:
                    det_emb = det_embs[i, :].copy()
                else:
                    det_emb = None
                detections_list.append(Detection(det_box, det_score, det_obj_conf, det_cls_conf, det_emb))

            # tracker.update(detections_list, vis_img.copy(), id_to_color, fid, visual_seq_pre_root)
            tracker.update(detections_list)

            tracklet_num = 0
            for tracklet in tracker.tracklets_list:
                if tracklet.is_confirmed() and tracklet.is_tracked():
                    if opt.visualise:
                        color = id_to_color[tracklet.tracklet_id]
                        vis_img = draw_bbox_id(vis_img, tracklet.tlbr, tracklet.tracklet_id, color)
                    bbox = tracklet.tlwh
                    results.append([fid + 1 - begin_fid, tracklet.tracklet_id, bbox[0], bbox[1], bbox[2], bbox[3]])
                    tracklet_num += 1

            if opt.visualise:
                text = f"Tracker: PE-Track  Frame: {fid + 1}"
                (_, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                vis_img = cv2.putText(vis_img, text, (20, text_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, lineType=cv2.LINE_AA)
                vis_img = cv2.putText(vis_img, f"num={tracklet_num}", (20, 2 * text_height + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, lineType=cv2.LINE_AA)
                cv2.imwrite(f'{visual_seq_root}/{fid + 1:08d}.jpg', vis_img)
        # break
        results_save_path = os.path.join(results_save_dir, f'{seq}.txt')
        write_results(results_save_path, results)
    
    if not opt.use_cache:
        mkdir(os.path.dirname(opt.save_cache_path))
        det_emb_model.dump_cache(opt.save_cache_path)

    if opt.mod == "val":
        os.system(f"python ./TrackEval/scripts/run_mot_challenge.py  \
                                            --SPLIT_TO_EVAL train  \
                                            --METRICS HOTA CLEAR Identity\
                                            --GT_FOLDER {opt.data_root}   \
                                            --SEQMAP_FILE {opt.val_map_path}  \
                                            --SKIP_SPLIT_FOL True   \
                                            --TRACKERS_TO_EVAL {opt.tracker_name} \
                                            --TRACKER_SUB_FOLDER tracker  \
                                            --USE_PARALLEL True  \
                                            --NUM_PARALLEL_CORES 8  \
                                            --PLOT_CURVES False   \
                                            --TRACKERS_FOLDER  {opt.save_root}  \
                                            --GT_LOC_FORMA {opt.val_type}")




if __name__ == '__main__':
    main()

