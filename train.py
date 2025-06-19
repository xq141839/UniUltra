import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import BinaryLoader
from loss import *
from tqdm import tqdm
import json
from model import SAM2
import hydra
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from sam2.build_sam import build_sam2
from torchmetrics.classification import BinaryAccuracy

torch.set_num_threads(8)
# matplotlib.use('TkAgg')

def train_model(model, criterion_mask, optimizer, scheduler, num_epochs=5):
    
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss_mask = []
            running_corrects_mask = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for _, img, labels, img_id in tqdm(dataloaders[phase]):      
                # wrap them in Variable
    
                img = Variable(img.cuda())
                labels = Variable(labels.cuda())
   
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_mask = model(x=img, gt=labels, img_id=img_id)
                pred_mask = torch.sigmoid(pred_mask)

                loss1 = criterion_mask(pred_mask, labels)
                score_mask1 = accuracy_metric(pred_mask, labels)
       

                loss = loss1

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss_mask.append(loss1.item())
                running_corrects_mask.append(score_mask1.item())
             

            epoch_loss = np.mean(running_loss_mask)
            epoch_acc = np.mean(running_corrects_mask)
            
            print('{} Loss 1: {:.4f} IoU 1: {:.4f} '.format(
                phase, np.mean(running_loss_mask), np.mean(running_corrects_mask)))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'outputs/sam2_adapter_point_{args.dataset}_{epoch}.pth')
            if phase == 'valid':
                scheduler.step()
                print(f"lr: {scheduler.get_last_lr()[0]}")
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    

    return Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='all', help='BUSI, DDTI, TN3K, UDIAT, TNBC')
    parser.add_argument('--sam_pretrain', type=str,default='/home/Qing_Xu/pretrain/sam2_hiera_large.pt', 
    help='pretrain/sam_vit_b_01ec64.pth, medsam_box_best_vitb.pth, medsam_vit_b, efficient_sam_vits, mobile_sam')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='epoches')
    parser.add_argument('--size', type=float, default=1.0, help='epoches')
    args = parser.parse_args()

    os.makedirs('outputs/', exist_ok=True)

    jsonfile1 = f'/home/Qing_Xu/IJCAI2025/FTSAM/datasets/BUSI/data_split.json'
    
    with open(jsonfile1, 'r') as f:
        df1 = json.load(f)
        train_size = int(round(len(df1['train']) * args.size, 0))
        print(f'BUSI: {train_size}')
        train_set1 = np.random.choice(df1['train'],train_size,replace=False)

    jsonfile2 = f'/home/Qing_Xu/IJCAI2025/FTSAM/datasets/DDTI/data_split.json'
    
    with open(jsonfile2, 'r') as f:
        df2 = json.load(f)
        train_size = int(round(len(df2['train']) * args.size, 0))
        print(f'DDTI: {train_size}')
        train_set2 = np.random.choice(df2['train'],train_size,replace=False)

    jsonfile3 = f'/home/Qing_Xu/IJCAI2025/FTSAM/datasets/TN3K/data_split.json'
    
    with open(jsonfile3, 'r') as f:
        df3 = json.load(f)
        train_size = len(df3['train'])
        train_set3 = np.random.choice(df3['train'],train_size,replace=False)
        print(f'TN3K: {train_size}')

    jsonfile4 = f'/home/Qing_Xu/IJCAI2025/FTSAM/datasets/UDIAT/data_split.json'
    
    with open(jsonfile4, 'r') as f:
        df4 = json.load(f)
        train_size = int(round(len(df4['train']) * args.size, 0))
        train_set4 = np.random.choice(df4['train'],train_size,replace=False)
        print(f'UDIAT: {train_size}')
    
    val_files = df2['test'] + df1['test'] + df3['test'] + df4['test']
    train_files = list(train_set2) + list(train_set1) + list(train_set3) + list(train_set4)

    train_dataset = BinaryLoader(args.dataset, train_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask'}))
    val_dataset = BinaryLoader(args.dataset, val_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ],
        additional_targets={'mask2': 'mask'}))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
    

    model_cfg = "sam2_hiera_l.yaml"
    # hydra is initialized on import of sam2, which sets the search path which can't be modified
    # so we need to clear the hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # reinit hydra with a new search path for configs
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    model = SAM2(build_sam2(model_cfg, args.sam_pretrain, mode='train'))

    # encoder_dict = torch.load(args.sam_pretrain)
    # pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    # model.load_state_dict(pre_dict, strict=False)

    # pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'prompt_encoder'}
    # model.load_state_dict(pre_dict, strict=False)

    # pre_dict = {k: v for k, v in encoder_dict.items() if list(k.split('.'))[0] == 'mask_decoder'}
    # model.load_state_dict(pre_dict, strict=False)

    # model.load_state_dict(torch.load(args.sam_pretrain), strict=True)
    # model.load_state_dict(torch.load(args.sam_pretrain, map_location={'cuda:4': 'cuda:0', 'cuda:6': 'cuda:0'}), strict=True)
    # model.load_state_dict(torch.load(args.sam_pretrain)["model"], strict=True)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    model = model.cuda()

    # for n, value in model.prompt_encoder.named_parameters():
    #     value.requires_grad = False

    for n, value in model.model.image_encoder.named_parameters():
        if f"edge" in n:
            if f"weight_h" in n or f"weight_v" in n:
                # print(n)
                value.requires_grad = False
            else:
                value.requires_grad = True
        else:
            value.requires_grad = False

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )

    print('Trainable Params = ' + str(trainable_params/1000**2) + 'M')

    total_params = sum(
	param.numel() for param in model.parameters()
    )

    print('Total Params = ' + str(total_params/1000**2) + 'M')

    print('Ratio = ' + str(trainable_params/total_params*100) + '%')

        
    # Loss, IoU and Optimizer
    mask_loss = BinaryMaskLoss() # nn.CrossEntropyLoss()
    accuracy_metric = BinaryIoU()#BinaryIoU()
    acc_metric = BinaryAccuracy(multidim_average='samplewise')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr = args.lr)
    # optimizer = optim.Adam(model.parameters(),lr = args.lr)
    # exp_lr_scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=200)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    # exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,min_lr=1e-7)
    Loss_list, Accuracy_list = train_model(model, mask_loss, optimizer, exp_lr_scheduler,
                           num_epochs=args.epoch)
    
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')