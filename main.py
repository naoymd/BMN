import os, sys
import shutil
import time, datetime
import pprint
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb
import yaml
import matplotlib.pyplot as plt
from addict import Dict
from bmn_dataset import ActivityNet_Captions_BMN_Dataset, Charades_STA_BMN_Dataset, TACoS_BMN_Dataset, BMN_Dataset, VideoDataSet, BMN_collate_fn
from model.bmn import BMN
from model.loss_fn import BMN_LossFunction
from model.post_bmn import post_BMN
from model.eval import eval_tIoU
from utils import sec2str, model_state_dict, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='two-stage TMR')
    parser.add_argument('config', type=str, help='path of a config file')
    parser.add_argument('--date', type=str, default='')
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        help='Add --no_wandb option if you do not want to use wandb.',
    )
    args = parser.parse_args()
    return args



def BMN_train(train_dataloader, val_dataloader, bmn, criterion, optimizer, lr_scheduler, CONFIG, args, device, date_path):
    train_start = time.time()
    
    # make result directory
    result_path = os.path.join(CONFIG.BMN_result_dir, date_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path, exist_ok=True)

    # make checkpoint directory
    checkpoint_path = CONFIG.BMN_checkpoint_dir
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    CONFIG_df = pd.DataFrame.from_dict(CONFIG, orient='index')
    CONFIG_df.to_csv(os.path.join(result_path, 'config.csv'), header=False)

    best_loss = 1e10
    train_loss_list = []
    train_pem_reg_loss_list = []
    train_pem_cls_loss_list = []
    train_tem_loss_list = []
    val_loss_list = []
    val_pem_reg_loss_list = []
    val_pem_cls_loss_list = []
    val_tem_loss_list = []
    lr_list = []
    for epoch in range(CONFIG.BMN_epoch_num):
        epoch_start = time.time()
        # train
        print('-'*5, 'train', '-'*5)
        bmn.train()
        train_loss = 0
        train_pem_reg_loss = 0
        train_pem_cls_loss = 0
        train_tem_loss = 0
        for i, train_data in enumerate(train_dataloader):
            input_data = train_data['video']
            gt_confidence_map = train_data['confidence_map']
            gt_start = train_data['start']
            gt_end = train_data['end']
            input_data = input_data.to(device)
            gt_confidence_map = gt_confidence_map.to(device)
            gt_start = gt_start.to(device)
            gt_end = gt_end.to(device)

            optimizer.zero_grad()
            
            confidence_map, start, end = bmn(input_data)
            loss = criterion(confidence_map, start, end, gt_confidence_map, gt_start, gt_end)
            loss[0].backward()
            train_loss += loss[0].cpu().detach().numpy()
            train_pem_reg_loss += loss[1].cpu().detach().numpy()
            train_pem_cls_loss += loss[2].cpu().detach().numpy()
            train_tem_loss += loss[3].cpu().detach().numpy()
            optimizer.step()

            if i % 200 == 0:
                print(epoch, i, loss[0])
            # break
        train_loss /= len(train_dataloader)
        train_pem_reg_loss /= len(train_dataloader)
        train_pem_cls_loss /= len(train_dataloader)
        train_tem_loss /= len(train_dataloader)
        train_loss_list.append(train_loss)
        train_pem_reg_loss_list.append(train_pem_reg_loss)
        train_pem_cls_loss_list.append(train_pem_cls_loss)
        train_tem_loss_list.append(train_tem_loss)
        print(sec2str(time.time() - epoch_start))

        # validation
        print('-'*5, 'validation', '-'*5)
        bmn.eval()
        val_loss = 0
        val_pem_reg_loss = 0
        val_pem_cls_loss = 0
        val_tem_loss = 0
        with torch.no_grad():
            for i, val_data in enumerate(val_dataloader):
                input_data = val_data['video']
                gt_confidence_map = val_data['confidence_map']
                gt_start = val_data['start']
                gt_end = val_data['end']
                input_data = input_data.to(device)
                gt_confidence_map = gt_confidence_map.to(device)
                gt_start = gt_start.to(device)
                gt_end = gt_end.to(device)

                confidence_map, start, end = bmn(input_data)
                loss = criterion(confidence_map, start, end, gt_confidence_map, gt_start, gt_end)
                
                val_loss += loss[0].cpu().detach().numpy()
                val_pem_reg_loss += loss[1].cpu().detach().numpy()
                val_pem_cls_loss += loss[2].cpu().detach().numpy()
                val_tem_loss += loss[3].cpu().detach().numpy()
        val_loss /= len(val_dataloader)
        val_pem_reg_loss /= len(val_dataloader)
        val_pem_cls_loss /= len(val_dataloader)
        val_tem_loss /= len(val_dataloader)
        val_loss_list.append(val_loss)
        val_pem_reg_loss_list.append(val_pem_reg_loss)
        val_pem_cls_loss_list.append(val_pem_cls_loss)
        val_tem_loss_list.append(val_tem_loss)

        save_checkpoint(checkpoint_path, epoch, bmn, optimizer, val_loss, lr_scheduler)
        if val_loss <= best_loss:
            best_loss = val_loss
            save_checkpoint(result_path, epoch, bmn, optimizer, val_loss, lr_scheduler)
        
        lr_list.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step(train_loss)
        
        epoch_end = time.time() - epoch_start
        print(
            'Epoch: [{}/{}], Time: {}, train_loss: {loss:.4f}, val_loss: {val_loss:.4f}'.format(
                epoch+1,
                CONFIG.BMN_epoch_num,
                sec2str(epoch_end),
                loss = train_loss,
                val_loss = val_loss
            )
        )
        print(
            'train: [pem_reg_loss: {}, pem_cls_loss: {}, tem_loss: {}]'.format(
                train_pem_reg_loss, train_pem_cls_loss, train_tem_loss
            )
        )

        # save BMN log
        log_dict = {
            'epoch': list(range(epoch+1)),
            'learning_rate': lr_list,
            'train_loss': train_loss_list,
            'train_pem_reg_loss': train_pem_reg_loss_list,
            'train_pem_cls_loss': train_pem_cls_loss_list,
            'train_tem_loss': train_tem_loss_list,
            'val_loss': val_loss_list,
            'val_pem_reg_loss': val_pem_reg_loss_list,
            'val_pem_cls_loss': val_pem_cls_loss_list,
            'val_tem_loss': val_tem_loss_list,
        }
        log_df = pd.DataFrame.from_dict(log_dict).set_index('epoch')
        log_df.to_csv(os.path.join(result_path, 'log.csv'), mode='w')
        plt.figure()
        plt.plot(train_loss_list, label='train')
        plt.plot(val_loss_list, label='val')
        plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(result_path, 'loss.png'))
        plt.close()
    
    # save figure of loss log
    train_end = time.time() - train_start
    print('finised train: {}'.format(sec2str(train_end)))


def BMN_test(test_dataloader, bmn, criterion, optimizer, lr_scheduler, CONFIG, args, device, date_path):
    # test
    test_start = time.time()
    print('-'*5, 'test', '-'*5)
    result_path = os.path.join(CONFIG.BMN_result_dir, date_path)
    csv_path = os.path.join(CONFIG.BMN_result_dir, CONFIG.dataset, 'csv')
    if os.path.exists(csv_path):
        shutil.rmtree(csv_path)
    os.makedirs(csv_path, exist_ok=True)
    bmn_dict = model_state_dict(result_path, 'checkpoint.pth')
    bmn.load_state_dict(bmn_dict['state_dict'])
    bmn.eval()
    with torch.no_grad():
        for i, test_data in enumerate(test_dataloader):
            video_id = test_data['video_id']
            input_data = test_data['video']
            gt_confidence_map = test_data['confidence_map']
            gt_start = test_data['start']
            gt_end = test_data['end']
            input_data = input_data.to(device)
            gt_confidence_map = gt_confidence_map.to(device)
            gt_start = gt_start.to(device)
            gt_end = gt_end.to(device)

            confidence_map, start, end = bmn(input_data)

            start_map = start.squeeze(dim=0).cpu().detach().numpy()
            end_map = end.squeeze(dim=0).cpu().detach().numpy() 
            cls_confidence_map = confidence_map.squeeze(dim=0)[0, :, :].cpu().detach().numpy()
            reg_confidence_map = confidence_map.squeeze(dim=0)[1, :, :].cpu().detach().numpy()

            proposals = []
            if CONFIG.dataset in ['ActivityNet', 'Charades-STA', 'TACoS']:
                max_video_length = CONFIG.max_video_length
            else:
                max_video_length = 100
            for i in range(max_video_length-1):
                for j in range(max_video_length-1):
                    start_index = i
                    end_index = j + 1
                    if start_index < end_index and end_index < max_video_length:
                        xmin = start_index / max_video_length
                        xmax = end_index / max_video_length
                        xmin_score = start_map[start_index]
                        xmax_score = end_map[end_index]
                        cls_score = cls_confidence_map[i, j]
                        reg_score = reg_confidence_map[i, j]
                        score = xmin_score * xmax_score * cls_score * reg_score
                        proposals.append(
                            [video_id[0], xmin, xmax, xmin_score, xmax_score, cls_score, reg_score, score]
                        )
            
            col_name = ['video_id', 'xmin', 'xmax', 'xmin_score', 'xmax_score', 'cls_score', 'reg_socre', 'score']
            proposals_df = pd.DataFrame(proposals, columns=col_name)
            proposals_df.to_csv(os.path.join(csv_path, video_id[0])+'.csv', index=False)
    
    print('BMN post processing')
    post_BMN(mode='val', **CONFIG)
    

def BMN_main(CONFIG, args, device, date_path):
    print('-'*10, 'BMN train and validation', '-'*10)
    # BMN_train loading dataset
    if CONFIG.dataset == 'ActivityNet':
        train_dataset = ActivityNet_Captions_BMN_Dataset(mode='train', **CONFIG)
        val_dataset = ActivityNet_Captions_BMN_Dataset(mode='val', **CONFIG)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    elif CONFIG.dataset == 'Charades-STA':
        train_dataset = Charades_STA_BMN_Dataset(mode='train', **CONFIG)
        val_dataset = Charades_STA_BMN_Dataset(mode='val', **CONFIG)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    elif CONFIG.dataset == 'TACoS':
        train_dataset = TACoS_BMN_Dataset(mode='train', **CONFIG)
        val_dataset = TACoS_BMN_Dataset(mode='val', **CONFIG)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    elif CONFIG.dataset == 'Original':
        train_dataset = VideoDataSet(mode='train', **CONFIG)
        val_dataset = VideoDataSet(mode='val', **CONFIG)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    else:
        train_dataset = BMN_Dataset(mode='train', **CONFIG)
        val_dataset = BMN_Dataset(mode='val', **CONFIG)
        train_dataloader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset = val_dataset,
            batch_size = CONFIG.BMN_batch_size,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    
    video_size = iter(train_dataloader).next()['video'].size()
    print('video_size:', video_size)
    print(len(train_dataset), len(val_dataset))
    print(len(train_dataloader), len(val_dataloader))

    
    bmn = BMN(video_size, **CONFIG)
    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        bmn = nn.DataParallel(bmn, device_ids=device_ids)
    bmn = bmn.to(device)

    if not args.no_wandb:
        # Magic
        wandb.watch(bmn, log='all')

    optimizer = optim.Adam(
        bmn.parameters(),
        lr = math.sqrt(float(CONFIG.BMN_learning_rate)),
        weight_decay = float(CONFIG.BMN_weight_decay)
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=math.sqrt(float(CONFIG.factor)),
        verbose=True,
        min_lr=math.sqrt(float(CONFIG.min_learning_rate)),
    )
    criterion = BMN_LossFunction(video_size, **CONFIG)

    BMN_train(train_dataloader, val_dataloader, bmn, criterion, optimizer, lr_scheduler, CONFIG, args, device, date_path)
    
    print('-'*10, 'BMN test', '-'*10)
    # BMN_test loading dataset
    if CONFIG.dataset == 'ActivityNet':
        test_dataset = ActivityNet_Captions_BMN_Dataset(mode='val test', **CONFIG)
        test_dataloader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            batch_size = 1,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    elif CONFIG.dataset == 'Charades-STA':
        test_dataset = Charades_STA_BMN_Dataset(mode='val test', **CONFIG)
        test_dataloader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            batch_size = 1,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    elif CONFIG.dataset == 'TACoS':
        test_dataset = TACoS_BMN_Dataset(mode='val test', **CONFIG)
        test_dataloader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            batch_size = 1,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    elif CONFIG.dataset == 'Original':
        test_dataset = VideoDataSet(mode='val', **CONFIG)
        test_dataloader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            batch_size = 1,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    else:
        test_dataset = BMN_Dataset(mode='val test', **CONFIG)
        test_dataloader = torch.utils.data.DataLoader(
            dataset = test_dataset,
            batch_size = 1,
            shuffle = False,
            drop_last = True,
            num_workers = CONFIG.num_workers,
            collate_fn = BMN_collate_fn
        )
    print(len(test_dataset))
    print(len(test_dataloader))
    BMN_test(test_dataloader, bmn, criterion, optimizer, lr_scheduler, CONFIG, args, device, date_path)


def main(date):
    # argparser
    args = parse_args()
    if args.date != '':
        date = args.date

    # configuration
    CONFIG = Dict(yaml.safe_load(open(args.config)))
    # pprint.pprint(CONFIG)

    # cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    if device == 'cpu':
        print('You have to use GPUs because training CNN is computationally expensive.')
        sys.exit(1)

    # weights and biases
    if not args.no_wandb:
        wandb.init(
            config=CONFIG, project='two-stage-Temporal Moment Retrieval', job_type='training',
        )
    
    # date path
    date_path = date
    # config_name = str(args.config)[14:-5]
    # date_path = os.path.join(date, config_name)

    BMN_main(CONFIG, args, device, date)

if __name__ == '__main__':
    start_main = time.time()
    start_now = datetime.datetime.now()
    # date = start_now.strftime('%Y-%m-%d/%H')
    # date = start_now.strftime('%Y-%m-%d')
    date = start_now.strftime('%Y-%m')
    print(start_now.strftime('%Y/%m/%d %H:%M:%S'))

    main(date)

    end_main = sec2str(time.time() - start_main)
    end_now = datetime.datetime.now()
    print(
        'Finished main.py! | {} | {}'.format(
        end_main, end_now.strftime('%Y/%m/%d %H:%M:%S'))
    )
    print('='*70)