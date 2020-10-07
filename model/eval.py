import os
import pandas as pd
import numpy as np
import torch


class eval_tIoU():
    def __init__(self, save_path, proposal_num=100):
        super().__init__()
        self.proposal_num = proposal_num
        self.save_path = save_path
        self.tIoU_highest = 0.9
        self.tIoU_higher = 0.7
        self.tIoU_high = 0.5
        self.tIoU_mid = 0.3
        self.tIoU_low = 0.1
        self.count = 0
        self.highest_count = 0
        self.higher_count = 0
        self.high_count = 0
        self.mid_count = 0
        self.low_count = 0
        self.no_count = 0
        self.idx_count = 0


    def counting(self, tIoU):
        self.count += 1
        if tIoU >= self.tIoU_highest:
            self.highest_count += 1
        elif tIoU >= self.tIoU_higher and tIoU < self.tIoU_highest:
            self.higher_count += 1
        elif tIoU >= self.tIoU_high and tIoU < self.tIoU_higher:
            self.high_count += 1
        elif tIoU >= self.tIoU_mid and tIoU < self.tIoU_high:
            self.mid_count += 1
        elif tIoU >= self.tIoU_low and tIoU < self.tIoU_mid:
            self.low_count += 1
        else:
            self.no_count += 1
        highest_count = self.highest_count
        higher_count = highest_count + self.higher_count
        high_count = higher_count + self.high_count
        mid_count = high_count + self.mid_count
        low_count = mid_count + self.low_count

        eval_dict = {
            'R{}@{}'.format(self.proposal_num, self.tIoU_highest): [100*highest_count/self.count],
            'R{}@{}'.format(self.proposal_num, self.tIoU_higher): [100*higher_count/self.count],
            'R{}@{}'.format(self.proposal_num, self.tIoU_high): [100*high_count/self.count],
            'R{}@{}'.format(self.proposal_num, self.tIoU_mid): [100*mid_count/self.count],
            'R{}@{}'.format(self.proposal_num, self.tIoU_low): [100*low_count/self.count],
            'R{}@no count'.format(self.proposal_num): [100*self.no_count/self.count],
        }

        eval_df = pd.DataFrame.from_dict(eval_dict)
        eval_df.to_csv(os.path.join(self.save_path, 'eval.csv'), mode='w')
        return eval_dict

    def evaluating(self, cos_sim, pred, gt):
        max_tIoU = 0
        max_tIoU_idx = 0
        max_cos_sim_idx = np.argmax(cos_sim)
        max_cos_sim_tIoU = self.temporal_IoU(pred[max_cos_sim_idx], gt)
        for i in range(self.proposal_num):
            tIoU = self.temporal_IoU(pred[i], gt)
            if tIoU >= max_tIoU:
                max_tIoU = tIoU
                max_tIoU_idx = i
        if max_tIoU_idx == max_cos_sim_idx:
            self.idx_count += 1
        eval_dict = self.counting(max_cos_sim_tIoU)
        eval_dict['idx_count'] = ['{} / {}'.format(self.idx_count, self.count)]

        eval_df = pd.DataFrame.from_dict(eval_dict)
        eval_df.to_csv(os.path.join(self.save_path, 'eval.csv'), mode='w')
        return eval_dict

    def temporal_IoU(self, timestamp_pred, timestamp_gt):
        union = (min(timestamp_pred[0], timestamp_gt[0]),
                max(timestamp_pred[1], timestamp_gt[1]))
        inter = (max(timestamp_pred[0], timestamp_gt[0]),
                min(timestamp_pred[1], timestamp_gt[1]))
        t_iou = 1.0*(inter[1]-inter[0])/max((union[1]-union[0]), 1)
        if t_iou < 0:
            t_iou = 0
        return t_iou