# -*- coding: utf-8 -*-
import glob
import h5py
import os, sys
import shutil
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
from model.eval import eval_tIoU


class post_BMN():
    def __init__(self, mode='test', **kwargs):
        super().__init__()
        # config.yaml作成後に編集
        self.dataset = kwargs.get('dataset', 'Charades-STA')
        if self.dataset == 'ActivityNet':
            self.video_path = kwargs.get('video_path', '../dataset/activitynet/C3D/c3d-feat')
            self.annotation_path = kwargs.get('annotation_path', '../dataset/activitynet/ActivityNet_Captions')
            annotation_file_list = kwargs.get('annotation_file_list', ['train.json', 'val_1.json', 'val_2.json'])
            self.max_video_length = kwargs.get('max_video_length', 1000)
            self.min_video_length = kwargs.get('min_video_length', 50)
        elif self.dataset == 'Charades-STA':
            self.video_path = kwargs.get('video_path', '../dataset/charades_sta/video-feat')
            self.annotation_path = kwargs.get('annotation_path', '../dataset/charades_sta/annotation')
            annotation_file_list = kwargs.get(
                'annotation_file_list',
                [('charades_sta_train.txt', 'Charades_v1_train.csv'),
                ('charades_sta_test.txt', 'Charades_v1_test.csv'),
                ('charades_sta_test.txt', 'Charades_v1_test.csv')]
                )
            self.max_video_length = kwargs.get('max_video_length', 300)
            self.min_video_length = kwargs.get('min_video_length', 100)
        elif self.dataset == 'TACoS':
            self.video_path = kwargs.get('video_path', '../dataset/tacos/video-feat')
            self.annotation_path = kwargs.get('annotation_path', '../dataset/tacos/annotation')
            annotation_file_list = kwargs.get('annotation_file_list', ['train.json', 'val.json', 'test.json'])
            self.max_video_length = kwargs.get('max_video_length', 1500)
            self.min_video_length = kwargs.get('min_video_length', 100)
        else:
            self.video_path = kwargs.get('video_path', '../dataset/activitynet/C3D/c3d-feat')
            self.annotation_path = kwargs.get('annotation_path', '../dataset/activitynet/ActivityNet_Captions')
            annotation_file_list = kwargs.get('annotation_file_list', ['train.json', 'val_1.json', 'val_2.json'])
            self.max_video_length = kwargs.get('max_video_length', 1000)
            self.min_video_length = kwargs.get('min_video_length', 50)
            self.feature_path = "../dataset/activitynet/activitynet_feature_cuhk/csv_mean_100"
                
        if mode == 'train':
            self.annotation_list = self._get_annotation(annotation_file_list[0])
        elif mode == 'val':
            self.annotation_list = self._get_annotation(annotation_file_list[1])
        elif mode == 'val test':
            val_annotation_list = self._get_annotation(annotation_file_list[1])
            test_annotation_list = self._get_annotation(annotation_file_list[2])
            self.annotation_list = val_annotation_list + test_annotation_list
        elif mode == 'test':
            self.annotation_list = self._get_annotation(annotation_file_list[2])
        elif mode == 'all':
            train_annotation_list = self._get_annotation(annotation_file_list[0])
            val_annotation_list = self._get_annotation(annotation_file_list[1])
            test_annotation_list = self._get_annotation(annotation_file_list[2])
            self.annotation_list = train_annotation_list + val_annotation_list + test_annotation_list
        print(len(self.annotation_list))

        result_dir = kwargs.get('BMN_result_dir', './BMN_result')
        self.csv_path = os.path.join(result_dir, self.dataset, 'csv')
        self.result_path = os.path.join(result_dir, self.dataset, 'result')

        self.alpha = kwargs.get('soft_nms_alpha', 0.4)
        self.low_th = kwargs.get('soft_nms_low_th', 0.5)
        self.high_th = kwargs.get('soft_nms_high_th', 0.9)
        self.proposal_num = kwargs.get('BMN_proposal_num', 100)
        self.eval = eval_tIoU(os.path.join(result_dir, self.dataset), self.proposal_num)
        self.video_post_process()


    def _get_annotation(self, annotation_file):
        if self.dataset == 'ActivityNet':
            print(os.path.join(self.annotation_path, annotation_file))
            annotation_df = pd.read_json(os.path.join(self.annotation_path, annotation_file))
            annotation_list = []
            for video_id in list(annotation_df.columns):
                timestamps = annotation_df[video_id]['timestamps']
                duration = annotation_df[video_id]['duration']
                duration = float(duration)
                video_features = np.load(os.path.join(self.video_path, video_id + '.npy')).astype(np.float32)
                video_length, _ = video_features.shape
                timestamps_list = []
                if video_length >= self.min_video_length and video_length <= self.max_video_length:
                    for timestamp in timestamps:
                        start, end = timestamp
                        start = float(start)
                        end = min(float(end), duration)
                        if start > end:
                            start, end = end, start
                        timestamps_list.append([start, end])
                    annotation_dict = {'video_id': video_id, 'duration': duration, 'video_length': video_length, 'timestamps': timestamps_list}
                    annotation_list.append(annotation_dict)
        elif self.dataset == 'Charades-STA':
            txt_file, csv_file = annotation_file
            print(os.path.join(self.annotation_path, txt_file))
            print(os.path.join(self.annotation_path, csv_file))
            txt_df = pd.read_table(os.path.join(self.annotation_path, txt_file), header=None)
            csv_df = pd.read_csv(os.path.join(self.annotation_path, csv_file))
            annotation_list = []
            video_id_list = []
            previous_video_id = ''
            for i, txt_line in enumerate(list(txt_df[0])):
                annotation, sentence = txt_line.split('##')
                video_id, start, end = annotation.split(' ')
                # duration = float(csv_df.loc[csv_df['id'].str.contains(video_id)]['length'].values)
                csv_line = csv_df[csv_df['id'].isin([video_id])]
                duration = float(csv_line['length'].values)
                start =float(start)
                end = min(float(end), duration)
                hdf5_file = h5py.File(os.path.join(self.video_path, 'vgg_rgb_features.hdf5'), 'r')
                video_features = torch.from_numpy(hdf5_file[video_id][:]).float()
                video_length, _ = video_features.size()
                if video_length >= self.min_video_length and video_length <= self.max_video_length:
                    if previous_video_id != video_id and previous_video_id!='':
                        annotation_list.append(annotation_dict)
                    if start > end:
                        start, end = end, start
                    timestamp = [start, end]
                    if video_id not in video_id_list:
                        video_id_list.append(video_id)
                        annotation_dict = {'video_id': video_id, 'timestamps': [timestamp], 'video_length': video_length, 'duration': duration}
                    else:
                        timestamps_list = annotation_dict['timestamps']
                        timestamps_list.append(timestamp)
                        annotation_dict = {'video_id': video_id, 'timestamps': timestamps_list, 'video_length': video_length, 'duration': duration}
                    if i+1 == len(list(txt_df[0])):
                        annotation_list.append(annotation_dict)
                    previous_video_id = video_id
        elif self.dataset == 'TACoS':
            print(os.path.join(self.annotation_path, annotation_file))
            annotation_df = pd.read_json(os.path.join(self.annotation_path, annotation_file))
            annotation_list = []
            for video_id in list(annotation_df.columns):
                timestamps = annotation_df[video_id]['timestamps']
                fps = annotation_df[video_id]['fps']
                num_frames = annotation_df[video_id]['num_frames']
                duration = float(num_frames) / float(fps)
                hdf5_file = h5py.File(os.path.join(self.video_path, 'tall_c3d_features.hdf5'), 'r')
                video_features = torch.from_numpy(hdf5_file[video_id][:]).float()
                video_length, _ = video_features.size()
                timestamps_list = []
                if video_length >= self.min_video_length and video_length <= self.max_video_length:
                    for timestamp in timestamps:
                        start, end = timestamp
                        start = float(start) / float(fps)
                        end = min(float(end) / float(fps), duration)
                        if start > end:
                            start, end = end, start
                        timestamp = [start, end]
                        timestamps_list.append(timestamp)
                    annotation_dict = {'video_id': video_id, 'duration': duration, 'video_length': video_length, 'timestamps': timestamps_list}
                    annotation_list.append(annotation_dict)
        else:
            print(os.path.join(self.annotation_path, annotation_file))
            annotation_df = pd.read_json(os.path.join(self.annotation_path, annotation_file))
            annotation_list = []
            feature_list = list(map(lambda x: x.replace(self.feature_path+'/', '').rstrip('.csv'), glob.glob(os.path.join(self.feature_path, '*.csv'))))
            for video_id in list(annotation_df.columns):
                timestamps = annotation_df[video_id]['timestamps']
                duration = annotation_df[video_id]['duration']
                duration = float(duration)
                video_features = np.load(os.path.join(self.video_path, video_id + '.npy')).astype(np.float32)
                video_length, _ = video_features.shape
                timestamps_list = []
                if video_length >= self.min_video_length and video_length <= self.max_video_length and video_id in feature_list:
                    for timestamp in timestamps:
                        start, end = timestamp
                        start = float(start)
                        end = min(float(end), duration)
                        if start > end:
                            start, end = end, start
                        timestamps_list.append([start, end])
                    annotation_dict = {'video_id': video_id, 'duration': duration, 'video_length': video_length, 'timestamps': timestamps_list}
                    annotation_list.append(annotation_dict)
        return annotation_list


    def soft_nms(self, df):
        df = df.sort_values(by='score', ascending=False)
        tstart = list(df['xmin'].values)
        tend = list(df['xmax'].values)
        tscore = list(df['score'].values)

        rstart = []
        rend = []
        rscore = []

        while len(tscore) > 1 and len(rscore) < self.proposal_num:
            max_index = tscore.index(max(tscore))
            tmp_iou_list = self.iou_with_anchors(
                np.array(tstart), np.array(tend),
                tstart[max_index], tend[max_index]
            )
            for idx in range(0, len(tscore)):
                if idx != max_index:
                    tmp_iou = tmp_iou_list[idx]
                    tmp_width = tend[max_index] - tstart[max_index]
                    if tmp_iou > self.low_th + (self.high_th - self.low_th) * tmp_width:
                        tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / self.alpha)
            
            rstart.append(tstart[max_index])
            rend.append(tend[max_index])
            rscore.append(tscore[max_index])
            tstart.pop(max_index)
            tend.pop(max_index)
            tscore.pop(max_index)

        new_df = pd.DataFrame()
        new_df['score'] = rscore
        new_df['xmin'] = rstart
        new_df['xmax'] = rend
        return new_df


    def segment_iou(self, target_segment, candidate_segments):
        """Compute the temporal intersection over union between a
        target segment and all the test segments.
        Parameters
        ----------
        target_segment : 1d array
            Temporal target segment containing [starting, ending] times.
        candidate_segments : 2d array
            Temporal candidate segments containing N x [starting, ending] times.
        Outputs
        -------
        tiou : 1d array
            Temporal intersection over union score of the N's candidate segments.
        """
        tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
        tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) + (target_segment[1] - target_segment[0]) - segments_intersection
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        tIoU = segments_intersection.astype(float) / segments_union
        return tIoU

    
    def wrapper_segment_iou(self,target_segments, candidate_segments):
        """Compute intersection over union btw segments
        Parameters
        ----------
        target_segments : ndarray
            2-dim array in format [m x 2:=[init, end]]
        candidate_segments : ndarray
            2-dim array in format [n x 2:=[init, end]]
        Outputs
        -------
        tiou : ndarray
            2-dim array [n x m] with IOU ratio.
        Note: It assumes that candidate-segments are more scarce that target-segments
        """
        if candidate_segments.ndim != 2 or target_segments.ndim != 2:
            raise ValueError('Dimension of arguments is incorrect')

        n, m = candidate_segments.shape[0], target_segments.shape[0]
        tIoUs = np.empty((n, m))
        for i in range(m):
            tIoUs[:, i] = self.segment_iou(target_segments[i,:], candidate_segments)
            max_tIoU = max(tIoUs[:, i])
            eval_dict = self.eval.counting(max_tIoU)
        return eval_dict


    def iou_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        """
        Compute jaccard score between a box and the anchors.
        """
        len_anchors = anchors_max - anchors_min
        inter_xmin = np.maximum(anchors_min, box_min)
        inter_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(inter_xmax - inter_xmin, 0.)
        union_len = len_anchors - inter_len + box_max - box_min
        # print(inter_len, union_len)
        jaccard = np.divide(inter_len, union_len)
        return jaccard

    
    def video_post_process(self):
        # if os.path.exists(self.result_path):
        #     shutil.rmtree(self.result_path)
        os.makedirs(self.result_path, exist_ok=True)
        for annotation_dict in self.annotation_list:
            video_id = annotation_dict['video_id']
            # print(video_id)
            duration = annotation_dict['duration']
            video_length = annotation_dict['video_length']
            df = pd.read_csv(os.path.join(self.csv_path, video_id+'.csv'))

            if len(df) > 1:
                df = self.soft_nms(df)
            
            df = df.sort_values(by='score', ascending=False)
            start_end_list = []
            score_list = []

            if self.dataset in ['ActivityNet', 'Charades-STA', 'TACoS']:
                coefficient = self.max_video_length * duration / video_length
            else:
                coefficient = 100 * duration / video_length

            for i in range(min(self.proposal_num, len(df))):
                score_list.append(df['score'].values[i])
                start = float(max(0, min(df['xmin'].values[i] * coefficient, duration)))
                end = float(min(max(0, df['xmax'].values[i] * coefficient), duration))
                start_end_list.append([start, end])

            df = pd.DataFrame()
            df['video_id'] = [video_id for _ in range(len(score_list))]
            df['score'] = score_list
            df['timestamp'] = start_end_list
            df['ranking'] = range(len(score_list))
            df = df.set_index('ranking')
            df.to_csv(os.path.join(self.result_path, video_id+'.csv'))

            timestamps = np.array(start_end_list)
            gt_timestamps = np.array(annotation_dict['timestamps'])
            eval_dict = self.wrapper_segment_iou(gt_timestamps, timestamps)
        print(eval_dict)

if __name__ == '__main__':
    p = post_BMN()

