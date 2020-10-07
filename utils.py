import sys, os
import numpy as np
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
from typing import List


def sec2str(sec):
    if sec < 60:
        return 'elapsed: {:02d}s'.format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return 'elapsed: {:02d}m{:02d}s'.format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return 'elapsed: {:02d}h{:02d}m{:02d}s'.format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return 'elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s'.format(dy, hr, min, sec)


def model_state_dict(save_dir, model_checkpoint):
    model_params = os.path.join(save_dir, model_checkpoint)
    state_dict = torch.load(model_params)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k[:]
        new_state_dict[name] = v
    return state_dict


def save_checkpoint(
    save_path, epoch, model, optimizer, best_loss,
    scheduler=None, add_epoch2name=False
):
    save_states = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
    }

    if scheduler is not None:
        save_states['scheduler'] = scheduler.state_dict()

    if add_epoch2name:
        torch.save(
            save_states,
            os.path.join(
                save_path, 'epoch{}_checkpoint.pth'.format(epoch))
        )
    else:
        torch.save(save_states, os.path.join(
            save_path, 'checkpoint.pth'))


def resume(result_path, model, optimizer, scheduler=None):
    resume_path = os.path.join(result_path, 'checkpoint.pth')
    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(
        resume_path, map_location=lambda storage, loc: storage)

    begin_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])

    # confirm whether the optimizer matches that of checkpoints
    # optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, model, optimizer, best_loss, scheduler


def argrelmax(input: np.ndarray, threshold: float = 0.7) -> List:
    """
    Calculate arguments of relative maxima.
    input: np.array. boundary probability maps distributerd in [0, 1]
    input shape is (T)
    ignore the peak whose value is under threshold
    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    input[input < threshold] = 0.0
    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (input[:-2] < input[1:-1]) & (input[2:] < input[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )
    peak_idx = np.where(peak)[0].tolist()
    return peak_idx


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
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