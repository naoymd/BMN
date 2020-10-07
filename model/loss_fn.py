import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BMN_LossFunction(nn.Module):
    def __init__(self, video_size, **kwargs):
        super().__init__()
        self.eps = float(kwargs.get('eps', 1e-8))
        self.video_length = video_size[1]
        self.bm_mask = self.get_mask()


    def get_mask(self):
        mask = np.zeros([self.video_length, self.video_length], np.float32)
        for i in range(self.video_length):
            for j in range(self.video_length):
                mask[i, j] = 1
        mask = torch.tensor(mask)
        mask = mask.to(device)
        return mask


    def forward(self, pred_bm, pred_start, pred_end, gt_iou_map, gt_start, gt_end):
        """
        pred_bm: (batch_size, 2, self.video_length, self.video_length)
        pred_start: (batch_size, self.video_length)
        pred_end: (batch_size, self.video_length)
        """
        pred_bm_reg = pred_bm[:, 0, :, :].contiguous()
        pred_bm_cls = pred_bm[:, 1, :, :].contiguous()

        gt_iou_map = gt_iou_map.float() * self.bm_mask

        pem_reg_loss = self._pem_reg_loss(pred_bm_reg, gt_iou_map)
        pem_cls_loss = self._pem_cls_loss(pred_bm_cls, gt_iou_map)
        tem_loss = self._tem_loss(pred_start, pred_end, gt_start, gt_end)

        loss = tem_loss + 10*pem_reg_loss + pem_cls_loss
        return loss, pem_reg_loss, pem_cls_loss, tem_loss

    
    def _pem_reg_loss(self, pred_score, gt_iou_map):
        u_hmask = (gt_iou_map > 0.7).float()
        u_mmask = ((gt_iou_map <= 0.7) & (gt_iou_map > 0.3)).float()
        u_lmask = ((gt_iou_map <= 0.3) & (gt_iou_map > 0.)).float()
        u_lmask = u_lmask * self.bm_mask

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = torch.tensor(np.random.rand(*gt_iou_map.shape)).float().cuda()
        u_smmask = u_mmask * u_smmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / num_l
        u_slmask = torch.tensor(np.random.rand(*gt_iou_map.shape)).float().cuda()
        u_slmask = u_lmask * u_slmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        weights = u_hmask + u_smmask + u_slmask

        loss = F.mse_loss(pred_score * weights, gt_iou_map * weights)
        loss = 0.5 * torch.sum(loss * torch.ones(*weights.shape).cuda()) / torch.sum(weights)
        return loss

    
    def _pem_cls_loss(self, pred_score, gt_iou_map):
        p_mask = (gt_iou_map > 0.9).float()
        n_mask = (gt_iou_map <= 0.9).float()
        n_mask = n_mask * self.bm_mask

        num_positive = torch.sum(p_mask)
        num_entries = num_positive + torch.sum(n_mask)
        ratio = num_entries / num_positive
        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio
        loss_pos = coef_1 * torch.log(pred_score + self.eps) * p_mask
        loss_neg = coef_0 * torch.log(1.0 - pred_score + self.eps) * n_mask
        loss = -1 * torch.sum(loss_pos + loss_neg) / num_entries
        return loss


    def _tem_loss(self, pred_start, pred_end, gt_start, gt_end):
        loss_start = self._bi_loss(pred_start, gt_start)
        loss_end = self._bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss


    def _bi_loss(self, pred_score, gt_label):
        pred_score = pred_score.reshape(-1)
        gt_label = gt_label.reshape(-1)
        p_mask = (gt_label >= 0.5).float()
        num_entries = len(p_mask)
        num_positive = torch.sum(p_mask)
        ratio = num_entries / num_positive
        coef0 = 0.5 * ratio / (ratio - 1)
        coef1 = 0.5 * ratio
        loss_pos = coef1 * torch.log(pred_score + self.eps) * p_mask
        loss_neg = coef0 * torch.log(1.0 - pred_score + self.eps) * (1.0 - p_mask)
        loss = -1 * torch.mean(loss_pos + loss_neg)
        return loss

