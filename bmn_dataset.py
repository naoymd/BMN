import sys, os
import time
import math
import ast
import glob
import json
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# sys.path.append(os.pardir)
from utils import sec2str, iou_with_anchors, ioa_with_anchors


def BMN_collate_fn(batch_dataset):
    output = {
        'video_id': [],
        'video': [],
        'video_length': [],
        'start': [],
        'end': [],
        'confidence_map': []
    }
    for dataset in batch_dataset:
        output['video_id'].append(dataset['video_id'])
        output['video'].append(dataset['video'])
        output['video_length'].append(dataset['video_length'])
        output['start'].append(dataset['start'])
        output['end'].append(dataset['end'])
        output['confidence_map'].append(dataset['confidence_map'])
    output['video'] = torch.stack(output['video'], dim=0)
    output['video_length'] = torch.stack(output['video_length'], dim=0)
    output['start'] = torch.stack(output['start'], dim=0)
    output['end'] = torch.stack(output['end'], dim=0)
    output['confidence_map'] = torch.stack(output['confidence_map'], dim=0)
    return output


class ActivityNet_Captions_BMN_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='all', **kwargs):
        super().__init__()
        # config.yaml作成後に編集
        self.video_path = kwargs.get('video_path', '../dataset/activitynet/C3D/c3d-feat')
        self.annotation_path = kwargs.get('annotation_path', '../dataset/activitynet/ActivityNet_Captions')
        annotation_file_list = kwargs.get('annotation_file_list', ['train.json', 'val_1.json', 'val_2.json'])
        self.max_video_length = kwargs.get('max_video_length', 1000)
        self.min_video_length = kwargs.get('min_video_length', 50)
        self.anchor_xmin = [(i - 0.5) / self.max_video_length for i in range(self.max_video_length)]
        self.anchor_xmax = [(i + 0.5) / self.max_video_length for i in range(self.max_video_length)]
        
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
        # print(self.annotation_list)

    def _get_annotation(self, annotation_file):
        print(os.path.join(self.annotation_path, annotation_file))
        annotation_df = pd.read_json(os.path.join(self.annotation_path, annotation_file))
        # print(annotation_df)
        annotation_list = []
        duration_list = []
        video_length_list = []
        for i, video_id in enumerate(list(annotation_df.columns)):
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
                    timestamp = [start, end]
                    timestamps_list.append(timestamp)
                annotation_dict = {'video_id': video_id, 'timestamps': timestamps_list, 'duration': duration}
                annotation_list.append(annotation_dict)
                duration_list.append(duration)
                video_length_list.append(video_length)
            # duration_list.append(duration)
            # video_length_list.append(video_length)
            # if i == 4:
            #     break
        print('duration: max {}s, min {}s'.format(max(duration_list), min(duration_list)))
        print('video_length: max {}, min {}'.format(max(video_length_list), min(video_length_list)))
        os.makedirs('./dataset/activitynet', exist_ok=True)
        plt.figure()
        plt.hist(duration_list, bins=100)
        plt.savefig(os.path.join('./dataset/activitynet', 'duration_hist_'+annotation_file[:-5]))
        plt.close()
        plt.figure()
        plt.hist(video_length_list, bins=100)
        plt.savefig(os.path.join('./dataset/activitynet', 'video_length_hist_'+annotation_file[:-5]))
        plt.close()
        return annotation_list

    def _get_video_features(self, video_id):
        video_features = np.load(os.path.join(self.video_path, video_id + '.npy')).astype(np.float32)
        video_features = torch.tensor(video_features)
        video_length, _ = video_features.size()
        pad_length = self.max_video_length - video_length
        if pad_length > 0:
            video_features = F.pad(
                video_features.permute(1, 0), (0, pad_length), mode='constant', value=0.).permute(1, 0)
        return video_features, video_length

    def _get_label(self, duration, timestamps, video_length):
        gt_bbox = []
        # print(duration)
        coefficient = float(video_length) / (float(self.max_video_length) * float(duration))
        for timestamp in timestamps:
            # print(timestamp)
            start = max(min(float(timestamp[0]) * coefficient, 1), 0)
            end = min(max(float(timestamp[1]) * coefficient, 0), 1)
            gt_bbox.append([start, end])
        gt_bbox = np.array(gt_bbox)
        # print('gt_bbox', gt_bbox.shape)
        # print(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # print('gt_xmins')
        # print(gt_xmins)
        # print('gt_xmaxs')
        # print(gt_xmaxs)
        gt_lens = gt_xmaxs - gt_xmins
        # print('gt_lens')
        # print(gt_lens)
        gt_len_small = 3. / self.max_video_length
        # print('gt_len_small')
        # print(gt_len_small)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        # print('gt_start_bboxs, gt_end_bboxs')
        # print(gt_start_bboxs)
        # print(gt_end_bboxs)

        confidence_map = np.zeros((self.max_video_length, self.max_video_length))
        # print('confidence_map', confidence_map.shape)
        for i in range(self.max_video_length):
            for j in range(i, self.max_video_length):
                confidence_map[i, j] = np.max(
                    iou_with_anchors(
                        i / self.max_video_length,
                        (j+1) / self.max_video_length,
                        gt_xmins, gt_xmaxs
                    )
                )
        confidence_map = torch.tensor(confidence_map)

        start_label_map = []
        end_label_map = []
        for xmin, xmax in zip(self.anchor_xmin, self.anchor_xmax):
            start_label_map.append(
                np.max(
                    ioa_with_anchors(xmin, xmax, gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])
                )
            )
            end_label_map.append(
                np.max(
                    ioa_with_anchors(xmin, xmax, gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])
                )
            )
        # print('start_label_map')
        # print(start_label_map)
        # print('end_label_map')
        # print(end_label_map)
        start_label_map = torch.tensor(start_label_map)
        end_label_map = torch.tensor(end_label_map)
        
        return start_label_map, end_label_map, confidence_map

    def __getitem__(self, index):
        # print('-'*50)
        # print('index', index)
        annotation = self.annotation_list[index]
        # print(annotation)
        # print(type(annotation))
        video_id = annotation['video_id']
        # print(video_id)
        # print(type(video_id))
        video_features, video_length = self._get_video_features(video_id)
        # print(video_features.size())
        # print(type(video_features))
        # print(video_length)
        # print(type(video_length))
        duration = annotation['duration']
        # print(duration)
        # print(type(duration))
        timestamps = annotation['timestamps']
        # print(timestamps[0])
        # print(type(timestamps[0]))
        start_label_map, end_label_map, confidence_map = self._get_label(duration, timestamps, video_length)
        if isinstance(video_length, int):
            video_length = torch.tensor(video_length)
        dataset = {
            'video_id': video_id,
            'video': video_features,
            'video_length': video_length,
            'start': start_label_map,
            'end': end_label_map,
            'confidence_map': confidence_map
        }
        return dataset

    def __len__(self):
        return len(self.annotation_list)


class Charades_STA_BMN_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='all', **kwargs):
        super().__init__()
        # config.yaml作成後に編集
        self.video_path = kwargs.get('video_path', '../dataset/charades_sta/video-feat')
        self.annotation_path = kwargs.get('annotation_path', '../dataset/charades_sta/annotation')
        annotation_txt_file_list = kwargs.get('annotation_txt_file_list', ['charades_sta_train.txt', 'charades_sta_test.txt'])
        annotation_csv_file_list = kwargs.get('annotation_csv_file_list', ['Charades_v1_train.csv', 'Charades_v1_test.csv'])
        self.max_video_length = kwargs.get('max_video_length', 300)
        self.min_video_length = kwargs.get('min_video_length', 100)
        self.anchor_xmin = [(i - 0.5) / self.max_video_length for i in range(self.max_video_length)]
        self.anchor_xmax = [(i + 0.5) / self.max_video_length for i in range(self.max_video_length)]
        
        if mode == 'train':
            self.annotation_list = self._get_annotation(annotation_txt_file_list[0], annotation_csv_file_list[0])
        elif mode == 'val':
            self.annotation_list = self._get_annotation(annotation_txt_file_list[1], annotation_csv_file_list[1])
        elif mode == 'val test':
            self.annotation_list = self._get_annotation(annotation_txt_file_list[1], annotation_csv_file_list[1])
        elif mode == 'test':
            self.annotation_list = self._get_annotation(annotation_txt_file_list[1], annotation_csv_file_list[1])
        elif mode == 'all':
            train_annotation_list = self._get_annotation(annotation_txt_file_list[0], annotation_csv_file_list[0])
            test_annotation_list = self._get_annotation(annotation_txt_file_list[1], annotation_csv_file_list[1])
            self.annotation_list = train_annotation_list + test_annotation_list
        # print(self.annotation_list)

    def _get_annotation(self, txt_file, csv_file):
        print(os.path.join(self.annotation_path, txt_file))
        print(os.path.join(self.annotation_path, csv_file))
        txt_df = pd.read_table(os.path.join(self.annotation_path, txt_file), header=None)
        csv_df = pd.read_csv(os.path.join(self.annotation_path, csv_file))
        annotation_list = []
        duration_list = []
        video_length_list = []
        video_id_list = []
        previous_video_id = ''
        for i, txt_line in enumerate(list(txt_df[0])):
            # print(i)
            # print(txt_line)
            annotation, sentence = txt_line.split('##')
            # print(annotation)
            video_id, start, end = annotation.split(' ')
            # print(video_id)
            # duration = float(csv_df.loc[csv_df['id'].str.contains(video_id)]['length'].values)
            csv_line = csv_df[csv_df['id'].isin([video_id])]
            duration = float(csv_line['length'].values)
            # print(duration)
            start = float(start)
            end = min(float(end), duration)
            # print(start, end)
            hdf5_file = h5py.File(os.path.join(self.video_path, 'vgg_rgb_features.hdf5'), 'r')
            video_features = torch.from_numpy(hdf5_file[video_id][:]).float()
            video_length, _ = video_features.size()
            # print(video_features.size())
            if video_length >= self.min_video_length and video_length <= self.max_video_length:
                if previous_video_id != video_id and previous_video_id!='':
                    annotation_list.append(annotation_dict)
                if start > end:
                    start, end = end, start
                timestamp = [start, end]
                if video_id not in video_id_list:
                    video_id_list.append(video_id)
                    annotation_dict = {'video_id': video_id, 'timestamps': [timestamp], 'duration': duration}
                else:
                    timestamps_list = annotation_dict['timestamps']
                    timestamps_list.append(timestamp)
                    annotation_dict = {'video_id': video_id, 'timestamps': timestamps_list, 'duration': duration}
                if i+1 == len(list(txt_df[0])):
                    annotation_list.append(annotation_dict)
                previous_video_id = video_id
                duration_list.append(duration)
                video_length_list.append(video_length)
            # duration_list.append(duration)
            # video_length_list.append(video_length)
            # if i == 10:
            #     print(i, 'break')
            #     break
        print('duration: max {}s, min {}s'.format(max(duration_list), min(duration_list)))
        print('video_length: max {}, min {}'.format(max(video_length_list), min(video_length_list)))
        os.makedirs('./dataset/charades_sta', exist_ok=True)
        plt.figure()
        plt.hist(duration_list, bins=100)
        plt.savefig(os.path.join('./dataset/charades_sta', 'duration_hist_'+txt_file[13:-4]))
        plt.figure()
        plt.hist(video_length_list, bins=100)
        plt.savefig(os.path.join('./dataset/charades_sta', 'video_length_hist_'+txt_file[13:-4]))
        plt.close()
        # print(annotation_list)
        return annotation_list

    def _get_video_features(self, video_id):
        hdf5_file = h5py.File(os.path.join(self.video_path, 'vgg_rgb_features.hdf5'), 'r')
        video_features = torch.from_numpy(hdf5_file[video_id][:]).float()
        video_length, _ = video_features.size()
        pad_length = self.max_video_length - video_length
        if pad_length > 0:
            video_features = F.pad(
                video_features.permute(1, 0), (0, pad_length), mode='constant', value=0.).permute(1, 0)
        return video_features, video_length

    def _get_label(self, duration, timestamps, video_length):
        gt_bbox = []
        # print(duration)
        coefficient = float(video_length) / (float(self.max_video_length) * float(duration))
        for timestamp in timestamps:
            # print(timestamp)
            start = max(min(float(timestamp[0]) * coefficient, 1), 0)
            end = min(max(float(timestamp[1]) * coefficient, 0), 1)
            gt_bbox.append([start, end])
        gt_bbox = np.array(gt_bbox)
        # print('gt_bbox', gt_bbox.shape)
        # print(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # print('gt_xmins')
        # print(gt_xmins)
        # print('gt_xmaxs')
        # print(gt_xmaxs)
        gt_lens = gt_xmaxs - gt_xmins
        # print('gt_lens')
        # print(gt_lens)
        gt_len_small = 3. / self.max_video_length
        # print('gt_len_small')
        # print(gt_len_small)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        # print('gt_start_bboxs, gt_end_bboxs')
        # print(gt_start_bboxs)
        # print(gt_end_bboxs)

        confidence_map = np.zeros((self.max_video_length, self.max_video_length))
        # print('confidence_map', confidence_map.shape)
        for i in range(self.max_video_length):
            for j in range(i, self.max_video_length):
                confidence_map[i, j] = np.max(
                    iou_with_anchors(
                        i / self.max_video_length,
                        (j+1) / self.max_video_length,
                        gt_xmins, gt_xmaxs
                    )
                )
        confidence_map = torch.tensor(confidence_map)

        start_label_map = []
        end_label_map = []
        for xmin, xmax in zip(self.anchor_xmin, self.anchor_xmax):
            start_label_map.append(
                np.max(
                    ioa_with_anchors(xmin, xmax, gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])
                )
            )
            end_label_map.append(
                np.max(
                    ioa_with_anchors(xmin, xmax, gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])
                )
            )
        # print('start_label_map')
        # print(start_label_map)
        # print('end_label_map')
        # print(end_label_map)
        start_label_map = torch.tensor(start_label_map)
        end_label_map = torch.tensor(end_label_map)
        
        return start_label_map, end_label_map, confidence_map

    def __getitem__(self, index):
        # print('-'*50)
        # print('index', index)
        annotation = self.annotation_list[index]
        # print(annotation)
        # print(type(annotation))
        video_id = annotation['video_id']
        # print(video_id)
        # print(type(video_id))
        video_features, video_length = self._get_video_features(video_id)
        # print(video_features.size())
        # print(type(video_features))
        # print(video_length)
        # print(type(video_length))
        duration = annotation['duration']
        # print(duration)
        # print(type(duration))
        timestamps = annotation['timestamps']
        # print(timestamps[0])
        # print(type(timestamps[0]))
        start_label_map, end_label_map, confidence_map = self._get_label(duration, timestamps, video_length)
        if isinstance(video_length, int):
            video_length = torch.tensor(video_length)
        dataset = {
            'video_id': video_id,
            'video': video_features,
            'video_length': video_length,
            'start': start_label_map,
            'end': end_label_map,
            'confidence_map': confidence_map
        }
        return dataset

    def __len__(self):
        return len(self.annotation_list)


class TACoS_BMN_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='all', **kwargs):
        super().__init__()
        # config.yaml作成後に編集
        self.video_path = kwargs.get('video_path', '../dataset/tacos/video-feat')
        self.annotation_path = kwargs.get('annotation_path', '../dataset/tacos/annotation')
        annotation_file_list = kwargs.get('annotation_file_list', ['train.json', 'val.json', 'test.json'])
        self.max_video_length = kwargs.get('max_video_length', 1500)
        self.min_video_length = kwargs.get('min_video_length', 100)
        self.anchor_xmin = [(i - 0.5) / self.max_video_length for i in range(self.max_video_length)]
        self.anchor_xmax = [(i + 0.5) / self.max_video_length for i in range(self.max_video_length)]
        
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
        # print(self.annotation_list)

    def _get_annotation(self, annotation_file):
        print(os.path.join(self.annotation_path, annotation_file))
        annotation_df = pd.read_json(os.path.join(self.annotation_path, annotation_file))
        # print(annotation_df)
        # print(annotation_df.columns)
        # print(annotation_df.index)
        annotation_list = []
        duration_list = []
        video_length_list = []
        for i, video_id in enumerate(list(annotation_df.columns)):
            # print(i)
            # print(video_id)
            # print(annotation_df[video_id])
            timestamps = annotation_df[video_id]['timestamps']
            fps = annotation_df[video_id]['fps']
            num_frames = annotation_df[video_id]['num_frames']
            duration = float(num_frames) / float(fps)
            # print(timestamps)
            # print(fps)
            # print(num_frames)
            hdf5_file = h5py.File(os.path.join(self.video_path, 'tall_c3d_features.hdf5'), 'r')
            video_features = torch.from_numpy(hdf5_file[video_id][:]).float()
            video_length, _ = video_features.size()
            # print(video_features.size())
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
                annotation_dict = {'video_id': video_id, 'timestamps': timestamps_list, 'duration': duration}
                annotation_list.append(annotation_dict)
                duration_list.append(duration)
                video_length_list.append(video_length)
            # duration_list.append(duration)
            # video_length_list.append(video_length)
            # if i == 0:
            #     print(i, 'break')
            #     break
        print('duration: max {}s, min {}s'.format(max(duration_list), min(duration_list)))
        print('video_length: max {}, min {}'.format(max(video_length_list), min(video_length_list)))
        # os.makedirs('./dataset/tacos', exist_ok=True)
        # plt.figure()
        # plt.hist(duration_list, bins=25)
        # plt.savefig(os.path.join('./dataset/tacos', 'duration_hist_'+annotation_file[:-5]))
        # plt.figure()
        # plt.hist(video_length_list, bins=25)
        # plt.savefig(os.path.join('./dataset/tacos', 'video_length_hist_'+annotation_file[:-5]))
        # plt.close()
        # print(annotation_list)
        return annotation_list

    def _get_video_features(self, video_id):
        hdf5_file = h5py.File(os.path.join(self.video_path, 'tall_c3d_features.hdf5'), 'r')
        video_features = torch.from_numpy(hdf5_file[video_id][:]).float()
        video_length, _ = video_features.size()
        pad_length = self.max_video_length - video_length
        if pad_length > 0:
            video_features = F.pad(
                video_features.permute(1, 0), (0, pad_length), mode='constant', value=0.).permute(1, 0)
        return video_features, video_length

    def _get_label(self, duration, timestamps, video_length):
        gt_bbox = []
        # print(duration)
        coefficient = float(video_length) / (float(self.max_video_length) * float(duration))
        for timestamp in timestamps:
            # print(timestamp)
            start = max(min(float(timestamp[0]) * coefficient, 1), 0)
            end = min(max(float(timestamp[1]) * coefficient, 0), 1)
            gt_bbox.append([start, end])
        gt_bbox = np.array(gt_bbox)
        # print('gt_bbox', gt_bbox.shape)
        # print(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # print('gt_xmins')
        # print(gt_xmins)
        # print('gt_xmaxs')
        # print(gt_xmaxs)
        gt_lens = gt_xmaxs - gt_xmins
        # print('gt_lens')
        # print(gt_lens)
        gt_len_small = 3. / self.max_video_length
        # print('gt_len_small')
        # print(gt_len_small)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        # print('gt_start_bboxs, gt_end_bboxs')
        # print(gt_start_bboxs)
        # print(gt_end_bboxs)

        confidence_map = np.zeros((self.max_video_length, self.max_video_length))
        # print('confidence_map', confidence_map.shape)
        for i in range(self.max_video_length):
            for j in range(i, self.max_video_length):
                confidence_map[i, j] = np.max(
                    iou_with_anchors(
                        i / self.max_video_length,
                        (j+1) / self.max_video_length,
                        gt_xmins, gt_xmaxs
                    )
                )
        confidence_map = torch.tensor(confidence_map)

        start_label_map = []
        end_label_map = []
        for xmin, xmax in zip(self.anchor_xmin, self.anchor_xmax):
            start_label_map.append(
                np.max(
                    ioa_with_anchors(xmin, xmax, gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])
                )
            )
            end_label_map.append(
                np.max(
                    ioa_with_anchors(xmin, xmax, gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])
                )
            )
        # print('start_label_map')
        # print(start_label_map)
        # print('end_label_map')
        # print(end_label_map)
        start_label_map = torch.tensor(start_label_map)
        end_label_map = torch.tensor(end_label_map)
        
        return start_label_map, end_label_map, confidence_map

    def __getitem__(self, index):
        # print('-'*50)
        # print('index', index)
        annotation = self.annotation_list[index]
        # print(annotation)
        # print(type(annotation))
        video_id = annotation['video_id']
        # print(video_id)
        # print(type(video_id))
        video_features, video_length = self._get_video_features(video_id)
        # print(video_features.size())
        # print(type(video_features))
        # print(video_length)
        # print(type(video_length))
        duration = annotation['duration']
        # print(duration)
        # print(type(duration))
        timestamps = annotation['timestamps']
        # print(timestamps[0])
        # print(type(timestamps[0]))
        start_label_map, end_label_map, confidence_map = self._get_label(duration, timestamps, video_length)
        if isinstance(video_length, int):
            video_length = torch.tensor(video_length)
        dataset = {
            'video_id': video_id,
            'video': video_features,
            'video_length': video_length,
            'start': start_label_map,
            'end': end_label_map,
            'confidence_map': confidence_map
        }
        return dataset

    def __len__(self):
        return len(self.annotation_list)



class BMN_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', **kwargs):
        super().__init__()
        # config.yaml作成後に編集
        self.video_path = kwargs.get('video_path', '../dataset/activitynet/C3D/c3d-feat')
        self.annotation_path = kwargs.get('annotation_path', '../dataset/activitynet/ActivityNet_Captions')
        annotation_file_list = kwargs.get('annotation_file_list', ['train.json', 'val_1.json', 'val_2.json'])
        self.max_video_length = kwargs.get('max_video_length', 1000)
        self.min_video_length = kwargs.get('min_video_length', 50)
        self.temporal_scale = 100
        self.feature_path = "../dataset/activitynet/activitynet_feature_cuhk/csv_mean_100"
        self.anchor_xmin = [(i - 0.5) / self.temporal_scale for i in range(self.temporal_scale)]
        self.anchor_xmax = [(i + 0.5) / self.temporal_scale for i in range(self.temporal_scale)]
        
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
        # print(self.annotation_list)

    def _get_annotation(self, annotation_file):
        print(os.path.join(self.annotation_path, annotation_file))
        annotation_df = pd.read_json(os.path.join(self.annotation_path, annotation_file))
        # print(annotation_df)
        annotation_list = []
        duration_list = []
        video_length_list = []
        feature_list = list(map(lambda x: x.replace(self.feature_path+'/', '').rstrip('.csv'), glob.glob(os.path.join(self.feature_path, '*.csv'))))
        for i, video_id in enumerate(list(annotation_df.columns)):
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
                    timestamp = [start, end]
                    timestamps_list.append(timestamp)
                annotation_dict = {'video_id': video_id, 'timestamps': timestamps_list, 'duration': duration}
                annotation_list.append(annotation_dict)
                duration_list.append(duration)
                video_length_list.append(video_length)
            # duration_list.append(duration)
            # video_length_list.append(video_length)
            # if i == 100:
            #     break
        print('duration: max {}s, min {}s'.format(max(duration_list), min(duration_list)))
        print('video_length: max {}, min {}'.format(max(video_length_list), min(video_length_list)))
        os.makedirs('./dataset/activitynet_', exist_ok=True)
        plt.figure()
        plt.hist(duration_list, bins=100)
        plt.savefig(os.path.join('./dataset/activitynet_', 'duration_hist_'+annotation_file[:-5]))
        plt.close()
        plt.figure()
        plt.hist(video_length_list, bins=100)
        plt.savefig(os.path.join('./dataset/activitynet_', 'video_length_hist_'+annotation_file[:-5]))
        plt.close()
        return annotation_list
    
    def _get_video_features(self, video_id):
        video_features = pd.read_csv(os.path.join(self.feature_path, video_id + '.csv')).astype(np.float32)
        video_features = torch.tensor(video_features.values)
        video_length, _ = video_features.size()
        return video_features, video_length

    def _get_label(self, duration, timestamps, video_length):
        gt_bbox = []
        # print(duration)
        coefficient = 1 / float(duration)
        for timestamp in timestamps:
            # print(timestamp)
            start = max(min(float(timestamp[0]) * coefficient, 1), 0)
            end = min(max(float(timestamp[1]) * coefficient, 0), 1)
            gt_bbox.append([start, end])
        gt_bbox = np.array(gt_bbox)
        # print('gt_bbox', gt_bbox.shape)
        # print(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        # print('gt_xmins')
        # print(gt_xmins)
        # print('gt_xmaxs')
        # print(gt_xmaxs)
        gt_lens = gt_xmaxs - gt_xmins
        # print('gt_lens')
        # print(gt_lens)
        gt_len_small = 3. / self.temporal_scale
        # print('gt_len_small')
        # print(gt_len_small)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        # print('gt_start_bboxs, gt_end_bboxs')
        # print(gt_start_bboxs)
        # print(gt_end_bboxs)

        confidence_map = np.zeros((self.temporal_scale, self.temporal_scale))
        # print('confidence_map', confidence_map.shape)
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                confidence_map[i, j] = np.max(
                    iou_with_anchors(
                        i / self.temporal_scale,
                        (j+1) / self.temporal_scale,
                        gt_xmins, gt_xmaxs
                    )
                )
        confidence_map = torch.tensor(confidence_map)

        start_label_map = []
        end_label_map = []
        for xmin, xmax in zip(self.anchor_xmin, self.anchor_xmax):
            start_label_map.append(
                np.max(
                    ioa_with_anchors(xmin, xmax, gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])
                )
            )
            end_label_map.append(
                np.max(
                    ioa_with_anchors(xmin, xmax, gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])
                )
            )
        # print('start_label_map')
        # print(start_label_map)
        # print('end_label_map')
        # print(end_label_map)
        start_label_map = torch.tensor(start_label_map)
        end_label_map = torch.tensor(end_label_map)
        
        return start_label_map, end_label_map, confidence_map

    def __getitem__(self, index):
        # print('-'*50)
        # print('index', index)
        annotation = self.annotation_list[index]
        # print(annotation)
        # print(type(annotation))
        video_id = annotation['video_id']
        # print(video_id)
        # print(type(video_id))
        video_features, video_length = self._get_video_features(video_id)
        # print(video_features.size())
        # print(type(video_features))
        # print(video_length)
        # print(type(video_length))
        duration = annotation['duration']
        # print(duration)
        # print(type(duration))
        timestamps = annotation['timestamps']
        # print(timestamps[0])
        # print(type(timestamps[0]))
        start_label_map, end_label_map, confidence_map = self._get_label(duration, timestamps, video_length)
        if isinstance(video_length, int):
            video_length = torch.tensor(video_length)
        dataset = {
            'video_id': video_id,
            'video': video_features,
            'video_length': video_length,
            'start': start_label_map,
            'end': end_label_map,
            'confidence_map': confidence_map
        }
        return dataset

    def __len__(self):
        return len(self.annotation_list)


class VideoDataSet(torch.utils.data.Dataset):
    def __init__(self, mode="train", **kwargs):
        self.temporal_scale = 100
        self.feature_path = "../dataset/activitynet/activitynet_feature_cuhk/csv_mean_100"
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = mode
        self.video_info_path =  kwargs.get('video_path', '../dataset/activitynet/activitynet_annotations/video_info_new.csv')
        self.video_anno_path =  kwargs.get('annotation_path', '../dataset/activitynet/activitynet_annotations/anet_anno_action.json')
        self._getDatasetDict()
        self.anchor_xmin = [self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)]

    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = self._load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def __getitem__(self, index):
        video_id = self.video_list[index]
        video_data = self._load_file(index)
        start_label_map, end_label_map, confidence_map = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
        dataset = {
            'video_id': video_id,
            'video': video_data,
            'video_length': torch.tensor(self.temporal_scale),
            'start': start_label_map,
            'end': end_label_map,
            'confidence_map': confidence_map
        }
        return dataset

    
    def _load_json(self, file):
        with open(file) as json_file:
            json_data = json.load(json_file)
            return json_data

    def _load_file(self, index):
        video_name = self.video_list[index]
        video_df = pd.read_csv(os.path.join(self.feature_path, video_name+'.csv'))
        video_data = video_df.values[:, :]
        video_data = torch.Tensor(video_data)
        video_data.float()
        return video_data

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        gt_iou_map = torch.Tensor(gt_iou_map)

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    start = time.time()
    my_dataset = ActivityNet_Captions_BMN_Dataset(mode='test')
    print(len(my_dataset))
    my_dataset = Charades_STA_BMN_Dataset(mode='test')
    print(len(my_dataset))
    my_dataset = TACoS_BMN_Dataset(mode='test')
    print(len(my_dataset))
    my_dataset = BMN_Dataset(mode='test')
    print(len(my_dataset))
    print(sec2str(time.time()-start))
    data_loader = torch.utils.data.DataLoader(
        my_dataset, batch_size=4,
        shuffle=False, drop_last=True,
        num_workers=0, collate_fn=BMN_collate_fn
        )
    print('='*70)
    print(len(data_loader))
    for i, data in enumerate(data_loader):
        if i == 1:
            print(data['video_id'])
            print(data['video'].size())
            print(data['video_length'].size())
            print(data['start'].size())
            # print(data['start'])
            print(data['end'].size())
            # print(data['end'])
            print(data['confidence_map'].size())
            break
    print(sec2str(time.time()-start))