import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import BinaryLoader
from skimage import measure, morphology
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from model import SAM2
from functools import partial
from scipy import ndimage as ndi
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric
from sam2.build_sam import build_sam2
import hydra


def hd_score(p, y):

    tmp_hd = compute_hausdorff_distance(p, y)
    tmp_hd = torch.mean(tmp_hd)

    return tmp_hd.item()

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth) 
        
        
        return dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='TN3K',type=str, help='BUSI, DDTI, TN3K, UDIAT, FUGC, JNU/head')
    parser.add_argument('--mode', default='2d',type=str, help='2d, 3d')
    parser.add_argument('--gt_path', default='mask_1024',type=str, help='mask_1024_c1, mask_1024_')
    parser.add_argument('--jsonfile', default='data_split.json',type=str, help='')
    parser.add_argument('--size', type=int, default=1024, help='epoches')
    parser.add_argument('--model',default='/home/Qing_Xu/pretrain/sam2_hiera_large.pt', type=str, help='/home/Qing_Xu/pretrain/sam-med2d_b.pth, medsam_box_best_vitb, sam_vit_b_01ec64, medsam_vit_b, sam-med2d_b')
    args = parser.parse_args()
    
    save_png = f'visual/{args.dataset}/oursL/'
    # save_png = f"outputs/"
    feature_path = f'feature/{args.dataset}/'

    os.makedirs(save_png,exist_ok=True)
    os.makedirs(feature_path, exist_ok=True)

    print(args.dataset)
    print("------------------------------------------")
    if args.mode == "2d":
        args.jsonfile = f'/home/***/FTSAM/datasets/{args.dataset}/data_split.json'
    
        with open(args.jsonfile, 'r') as f:
            df = json.load(f)

        test_files = df['test']




    test_dataset = BinaryLoader(args.dataset, test_files, A.Compose([
                                        A.Resize(args.size, args.size),
                                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ToTensor()
                                        ], additional_targets={'mask2': 'mask'}))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)
    model_cfg = "sam2_hiera_l.yaml"
    # hydra is initialized on import of sam2, which sets the search path which can't be modified
    # so we need to clear the hydra instance
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # reinit hydra with a new search path for configs
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    model = SAM2(build_sam2(model_cfg, args.model, mode=None))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = nn.DataParallel(model)

    total_params = sum(
	param.numel() for param in model.parameters()
    )

    print('Total Params = ' + str(total_params/1000**2) + 'M')


    model.load_state_dict(torch.load(f'/home/Qing_Xu/hd1/xq/IJCAI2025/medsam2/outputs/sam2_adapter_final_best.pth'), strict=True)

    # model.load_state_dict(torch.load(f'/home/***/medsam2/outputs/small_ultra_20.pth'), strict=True)
    # pretrain_dict = torch.load('/home/***/medsam2/outputs/sam2_adapter_final_best.pth')
    # prompt_dict = {'.'.join(list(k.split('.'))[2:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[1] == 'sam_prompt_encoder'}
    # dec_dict = {'.'.join(list(k.split('.'))[2:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[1] == 'sam_mask_decoder'}
    # model.model.sam_prompt_encoder.load_state_dict(prompt_dict, strict=True) 
    # model.model.sam_mask_decoder.load_state_dict(dec_dict, strict=True) 

    # pretrain_dict = torch.load('/home/***/medsam2/outputs/small_ultra_18.pth')
    # selected_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'image_encoder'}
    # neck_dict = {'.'.join(list(k.split('.'))[1:]): v for k, v in pretrain_dict.items() if list(k.split('.'))[0] == 'neck'}
    # model.image_encoder.load_state_dict(selected_dict, strict=True)
    # model.neck.load_state_dict(neck_dict, strict=True) 
    
    # model.load_state_dict(torch.load(args.model)["model"], strict=True) #med2d
    # model.load_state_dict(torch.load(args.model, map_location={'cuda:4': 'cuda:0', 'cuda:6': 'cuda:0'}), strict=True)
    # model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model).items()}, strict=True)
    model = model.cuda()
    feature_npy = np.zeros((len(test_files), 256, 256))
    idx = 0
    
    TestAcc = Accuracy()
    TestPrecision = Precision()
    TestDice = Dice()
    TestRecall = Recall()
    TestF1 = F1(2)
    TestIoU = IoU()

    mIoU = []
    Accuracy = []
    Precision = []
    Recall = []
    F1_score = []
    DSC = []
    FPS = []
    image_ids = []
    hd_list = []
    
    since = time.time()

    model.train(False)  
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _, img, mask, img_id in tqdm(test_loader):
        # for _, img, mask, img_id in test_loader:
            print(img_id)
            img = Variable(img).cuda()            
            mask = Variable(mask).cuda()

            torch.cuda.synchronize()
            start = time.time()

            mask_pred = model(x=img, gt=mask, img_id=img_id)
            # mask_pred = mask_pred.squeeze(2)


            torch.cuda.synchronize()
            end = time.time()
            FPS.append(end-start)

            mask_pred = torch.sigmoid(mask_pred)
            # save_feature = lrmask.clone().cpu().detach().numpy()[0][0]
      
            # feature_npy[idx,:,:] = save_feature
            idx = idx + 1

            mask_pred[mask_pred >= 0.5] = 1
            mask_pred[mask_pred < 0.5] = 0

            # for box prompt
            if len(mask_pred.shape) > 4:
                mask_pred = torch.sum(mask_pred,dim=1, keepdim=True)#.squeeze(1)
                mask_pred = torch.squeeze(mask_pred, dim=1)
                mask_pred = torch.sum(mask_pred,dim=1, keepdim=True)

            mask_draw = mask_pred.clone().detach()
            gt_draw = mask.clone().detach()
            

            IoU = TestIoU(mask_pred,mask)
            dsc = TestDice(mask_pred,mask)
            hdscore = hd_score(mask_pred,mask)

            mask_pred = mask_pred.view(-1)
            mask = mask.view(-1)


            img_id = list(img_id[0].split('.'))[0]
            mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
            mask_numpy[mask_numpy==1] = 255 
            cv2.imwrite(f'{save_png}{img_id}.png',mask_numpy)

            mask_gt = gt_draw.cpu().detach().numpy()[0][0]
            mask_gt[mask_gt > 0] = 255 
            cv2.imwrite(f'{save_png}{img_id}_gt.png',mask_gt)

            accuracy = TestAcc(mask_pred.cpu(),mask.cpu())
            precision = TestPrecision(mask_pred.cpu(),mask.cpu())
            recall = TestRecall(mask_pred.cpu(),mask.cpu())
            f1score = TestF1(mask_pred.cpu(),mask.cpu())
            
            mIoU.append(IoU.item())
            DSC.append(dsc.item())
            if hdscore != float("inf"):
                hd_list.append(hdscore)
            Accuracy.append(accuracy.item())
            Precision.append(precision.item())
            Recall.append(recall.item())
            F1_score.append(f1score.item())
            image_ids.append(img_id)
            torch.cuda.empty_cache()

            # break
 
            
    time_elapsed = time.time() - since
    # np.save(f'{feature_path}{args.dataset}.npy',feature_npy)

    result_dict = {'image_id':image_ids, 'miou':mIoU, 'dice':DSC}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f'{save_png}results.csv',index=False)
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    F1 = 2 * np.mean(Precision) * np.mean(Recall) / (np.mean(Precision) + np.mean(Recall))
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    FPS.pop(0)
    print('FPS: {:.2f}'.format(1.0/(sum(FPS)/len(FPS))))
    print('mean IoU:',round(np.mean(mIoU),4),round(np.std(mIoU),4))
    print('mean accuracy:',round(np.mean(Accuracy),4),round(np.std(Accuracy),4))
    print('mean Precision:',round(np.mean(Precision),4))
    print('mean Recall:',round(np.mean(Recall),4))
    print('mean F1:',round(np.mean(F1),4))
    print('mean HD:',round(np.mean(hd_list),4),round(np.std(hd_list),4))
    print('mean Dice:',round(np.mean(DSC),4),round(np.std(DSC),4))

