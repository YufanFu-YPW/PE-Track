import os
import numpy as np
import pandas as pd
# import h5py
import pickle
from mot20_split_label import MOT20_train_seqs




def mkdirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def is_continuous(column):
    unique_values = np.unique(column)
    return np.all(np.diff(unique_values) == 1)


def trajectory_interpolation(data):
    df = pd.DataFrame(data, columns=['frame_index', 'track_id', 'x1', 'y1', 'width', 'height', 'vis'])
    df = df.sort_values(by='frame_index')
    full_index = np.arange(df['frame_index'].min(), df['frame_index'].max() + 1)
    complete_df = pd.DataFrame({'frame_index': full_index})
    complete_df = complete_df.merge(df, on='frame_index', how='left')
    complete_df[['track_id', 'x1', 'y1', 'width', 'height']] = complete_df[['track_id', 'x1', 'y1', 'width', 'height']].interpolate()
    complete_df.fillna(value=0.0, inplace=True)
    return complete_df.to_numpy()


def get_space_conditions(gt, num_frames):
    # fid, tid, cx, cy, w, h, vis
    space_conditions = {}  # Store spatial interaction information for each frame and start from the second frame {fid : condition}
    for fid in range(1, num_frames):
        f1_gt = gt[gt[:,0] == fid]
        f2_gt = gt[gt[:,0] == fid+1]
        space_conditions[fid+1] = _get_condition(f1_gt, f2_gt)
    return space_conditions

def _get_condition(f1_gt, f2_gt):
    space_condition = []
    for i in f2_gt:
        tid = i[1]
        j = f1_gt[f1_gt[:,1] == tid]
        if j.size != 0:
            j = j[0]
            delta = i[2:6] - j[2:6]
        else:
            delta = np.zeros(4)
        delta_tid = np.hstack((i[2:6], delta))
        space_condition.append(delta_tid)
    return np.vstack(space_condition)


def sample_MOT20(dataset_root, deal_mod, sample_length, save_root):
    for mod in deal_mod:
        print(f'Start processing {mod} datasets')
        mkdirs(save_root)
        save_path = os.path.join(save_root,'mot20_' + mod + f'_{sample_length}.pkl')
        # print(save_path)
        # break
        seq_root = os.path.join(dataset_root, 'train')
        # seqs = [s for s in os.listdir(seq_root)]
        seqs = MOT20_train_seqs

        sample_dataset = []
        for seq in seqs:
            print(seq + " : ")
            seq_info = open(os.path.join(seq_root, seq, 'seqinfo.ini')).read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

            if mod == 'train':
                gt_txt = os.path.join(seq_root, seq, 'gt', 'gt.txt')
            else:
                gt_txt = os.path.join(seq_root, seq, 'gt', f'gt_{mod}.txt')

            gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
            # fid, tid, x, y, w, h, mark, cls, vis
            gt = gt[gt[:,6] != 0]  # mark != 0
            # gt = gt[gt[:,7] == 1]  # cls == 1
            # fid, tid, x, y, w, h, vis
            gt = gt[:, [0,1,2,3,4,5,8]]
            # fid, tid, x, y, w, h, vis --> fid, tid, cx, cy, w, h, vis
            gt[:,2] += gt[:,4]/2
            gt[:,3] += gt[:,5]/2
            # 归一化
            gt[:,2] /= seq_width
            gt[:,4] /= seq_width
            gt[:,3] /= seq_height
            gt[:,5] /= seq_height

            num_tids = int(max(gt[:,1]))
            seq_tid_gt = {}  # {tid : tid_gt}
            for tid in range(0, num_tids + 1):
                tid_gt = gt[gt[:,1] == tid]
                # Interpolate discontinuous frames
                if not is_continuous(tid_gt[:,0]):
                    tid_gt = trajectory_interpolation(tid_gt)
                seq_tid_gt[tid] = tid_gt

            new_gt = np.vstack(list(seq_tid_gt.values()))
            num_frames = int(max(new_gt[:,0]))

            space_conditions = get_space_conditions(new_gt, num_frames)  # {fid : condition}

            # Trajectory sampling
            seq_sample_num = 0
            for tid, tid_gt in seq_tid_gt.items():
                tid_gt_len = len(tid_gt)
                print(f'tracklet {tid} lenght:{tid_gt_len}')
                if tid_gt_len <= sample_length + 1:
                    continue
                else:
                    num_sample_items = tid_gt_len - sample_length - 1
                    seq_sample_num += num_sample_items
                    print(f'tracklet {tid}: Number of samples = {num_sample_items}')
                    for i in range(num_sample_items):
                        box_1 = tid_gt[[i,i+1,i+2,i+3,i+4], 2:6]
                        box_2 = tid_gt[[i+1,i+2,i+3,i+4,i+5], 2:6]
                        delta_box = box_2 - box_1
                        long_history = np.hstack((box_2, delta_box))
                        label = tid_gt[i+6, 2:6]
                        delta_label = tid_gt[i+6, 2:6] - tid_gt[i+5, 2:6]
                        # space_frame = int(tid_gt[i+5, 0])
                        short_space = space_conditions[int(tid_gt[i+5, 0])]
                        sample_item = {'long_history': long_history,
                                       'short_space': short_space,
                                       'delta_label': delta_label,
                                       'label': label}
                        sample_dataset.append(sample_item)
            print(f'seq:{seq}  Total number of samples：{seq_sample_num}')

        print(f'DanceTrack {mod} datasets :  number of samples = {len(sample_dataset)}')
        with open(save_path, 'wb') as f:
            pickle.dump(sample_dataset, f)
        print(f'save to : {save_path}')





if __name__ == "__main__":
    sample_length = 5
    dataset_root = 'DataSets/MOT20'
    deal_mod = ['train']
    save_root = 'sample_datasets/MOT20'
    sample_MOT20(dataset_root, deal_mod, sample_length, save_root)

# len=5 total=360565

