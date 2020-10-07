# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn



class BMN(nn.Module):
    def __init__(self, video_size, **kwargs):
        super().__init__()
        batch_size = kwargs.get('BMN_batch_size', 8)
        self.video_length = video_size[1]
        input_size = video_size[2]
        # seq_length = kwargs.get('max_seq_length', 100)
        # kernel_size = kwargs.get('kernel_size', 5)
        # stride = kwargs.get('stride', 2)
        # dilation = kwargs.get('dilation', 1)
        # padding = kwargs.get('padding', 1)
        groups = kwargs.get('bmn_groups', 4)
        self.prop_boundary_ratio = kwargs.get('prop_boundary_ratio', 0.5)
        self.num_sample = kwargs.get('num_sample', 32)
        self.num_sample_perbin = kwargs.get('num_sample_perbin', 3)

        self.hidden_1d = kwargs.get('hidden_1d', 256)
        self.hidden_2d = kwargs.get('hidden_2d', 128)
        self.hidden_3d = kwargs.get('hidden_3d', 512)

        self.sample_mask = self._get_interp1d_mask()

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(input_size, self.hidden_1d, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_1d, self.hidden_1d, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_1d, self.hidden_1d, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_1d, self.hidden_1d, kernel_size=3, padding=1, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_1d, self.hidden_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_1d, self.hidden_3d, kernel_size=(self.num_sample, 1, 1), stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_3d, self.hidden_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_2d, self.hidden_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_2d, self.hidden_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p=kwargs.get('bmn_dropout', 0.1))


    def forward(self, x):
        """
        Input:
            x: (batch_size, self.video_length, input_size)
        Output:
            confidence_map: (batch_size, 2, self.video_length, self.video_length)
            start: (batch_size, self.video_length)
            end: (batch_size, self.video_length)
        """
        base_feature = self.x_1d_b(x.permute(0, 2, 1))
        base_feature = self.dropout(base_feature)
        start = self.x_1d_s(base_feature).squeeze(dim=1)
        end = self.x_1d_e(base_feature).squeeze(dim=1)
        confidence_map = self.x_1d_p(base_feature)
        confidence_map = self._boundary_matching_layer(confidence_map)
        confidence_map = self.x_3d_p(confidence_map).squeeze(dim=2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end


    def _boundary_matching_layer(self, x):
        input_size = x.size()
        x = torch.matmul(x, self.sample_mask)
        out = x.reshape(input_size[0], input_size[1], self.num_sample, self.video_length, self.video_length)
        return out


    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (self.num_sample * self.num_sample_perbin - 1.0)
        total_sample = [
            seg_xmin + plen_sample * ii
            for ii in range(self.num_sample * self.num_sample_perbin)
        ]
        p_mask = []
        for idx in range(self.num_sample):
            bin_samples = total_sample[idx * self.num_sample_perbin : (idx + 1) * self.num_sample_perbin]
            bin_vector = torch.zeros([self.video_length])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (self.video_length - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (self.video_length - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / self.num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = torch.stack(p_mask, dim=1)
        # print('p_mask', p_mask.size()) # (self.video_length, self.num_sample)
        return p_mask

    
    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.video_length):
            mask_mat_vector = []
            for start_index in range(self.video_length):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(sample_xmin, sample_xmax)
                    # p_mask = self._get_interp1d_bin_mask(sample_xmin, sample_xmax).to(device)
                else:
                    p_mask = torch.zeros([self.video_length, self.num_sample])
                    # p_mask = torch.zeros([self.video_length, self.num_sample]).to(device)
                mask_mat_vector.append(p_mask)
            mask_mat_vector = torch.stack(mask_mat_vector, dim=2)
            # print('mask_mat_vector', mask_mat_vector.size()) # (self.video_length, self.num_sample, self.video_length)
            mask_mat.append(mask_mat_vector)
        mask_mat = torch.stack(mask_mat, dim=3)
        # print('mask_mat', mask_mat.size()) # (self.video_length, self.num_sample, self.video_length, self.video_length)
        mask_mat = mask_mat.float()
        sample_mask = nn.Parameter(mask_mat.reshape(self.video_length, -1), requires_grad=False)
        # print(sample_mask.size())
        # print(sample_mask)
        return sample_mask


# class BMN_mask(nn.Module):
#     def __init__(self, video_size, **kwargs):
#         super().__init__()
#         self.video_length = video_size[1]
#         input_size = video_size[2]
#         self.prop_boundary_ratio = kwargs.get('prop_boundary_ratio', 0.5)
#         self.num_sample = kwargs.get('num_sample', 32)
#         self.num_sample_perbin = kwargs.get('num_sample_perbin', 3)


#     def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax):
#         # generate sample mask for a boundary-matching pair
#         plen = float(seg_xmax - seg_xmin)
#         plen_sample = plen / (self.num_sample * self.num_sample_perbin - 1.0)
#         total_sample = [
#             seg_xmin + plen_sample * ii
#             for ii in range(self.num_sample * self.num_sample_perbin)
#         ]
#         p_mask = []
#         for idx in range(self.num_sample):
#             bin_samples = total_sample[idx * self.num_sample_perbin : (idx + 1) * self.num_sample_perbin]
#             bin_vector = torch.zeros([self.video_length])
#             for sample in bin_samples:
#                 sample_upper = math.ceil(sample)
#                 sample_decimal, sample_down = math.modf(sample)
#                 if int(sample_down) <= (self.video_length - 1) and int(sample_down) >= 0:
#                     bin_vector[int(sample_down)] += 1 - sample_decimal
#                 if int(sample_upper) <= (self.video_length - 1) and int(sample_upper) >= 0:
#                     bin_vector[int(sample_upper)] += sample_decimal
#             bin_vector = 1.0 / self.num_sample_perbin * bin_vector
#             p_mask.append(bin_vector)
#         p_mask = torch.stack(p_mask, dim=1)
#         return p_mask

    
#     def forward(self):
#         # generate sample mask for each point in Boundary-Matching Map
#         mask_mat = []
#         for end_index in range(self.video_length):
#             mask_mat_vector = []
#             for start_index in range(self.video_length):
#                 if start_index <= end_index:
#                     p_xmin = start_index
#                     p_xmax = end_index + 1
#                     center_len = float(p_xmax - p_xmin) + 1
#                     sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
#                     sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
#                     p_mask = self._get_interp1d_bin_mask(sample_xmin, sample_xmax)
#                 else:
#                     p_mask = torch.zeros([self.video_length, self.num_sample])
#                 mask_mat_vector.append(p_mask)
#             mask_mat_vector = torch.stack(mask_mat_vector, dim=2)
#             mask_mat.append(mask_mat_vector)
#         mask_mat = torch.stack(mask_mat, dim=3)
#         mask_mat = mask_mat.float()
#         sample_mask = nn.Parameter(mask_mat.reshape(self.video_length, -1), requires_grad=False)
#         print(sample_mask.size())
#         # print(sample_mask)
#         return sample_mask


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input=torch.randn(2,100,500)
    video_size = input.size()
    device_ids = list(range(torch.cuda.device_count()))
    print(device_ids)
    device_id = torch.device('cuda:{}'.format(device_ids[-1]) if torch.cuda.is_available() else 'cpu')
    bmn = BMN(video_size)
    bmn = nn.DataParallel(bmn, device_ids=device_ids)
    bmn = bmn.to(device)
    a, b, c = bmn(input.to(device))
    print(a.shape,b.shape,c.shape)