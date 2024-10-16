import os
import argparse
import time
import random
import warnings
import json
import yaml

import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torch.autograd import Variable

from datasets.lamps_test import FaceForensics 
from datasets.factory import create_data_transforms
from model.LVNet_cls import Two_Stream_Net_Cls 
from utils.utils import *
import torchvision.utils as vutils

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default='./config/FF++.yml', type=str, help="Path to option YAML file.")
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--device', default=None, type=int, help='GPU id to use.')
    
    args = parser.parse_args()

    opt = yaml.safe_load(open(args.opt, 'r'))
    seed = opt["test"]["manual_seed"]
    
    if seed is not None:
        torch.cuda.manual_seed_all(seed)

    main_worker(args, opt)

def main_worker(args, opt):
    torch.cuda.set_device(args.gpu)
    args.device = torch.device(args.gpu)

    print(f"Creating model: {opt['model']['baseline']}")
    model = Two_Stream_Net_Cls()  
    model.to(args.device)

    state_dict = torch.load(opt['test']['ckt_path'], map_location='cpu')
    model.load_state_dict(cleanup_state_dict(state_dict['state_dict']), strict=False)

    criterion = nn.CrossEntropyLoss().to(args.device)

    all_transform = create_data_transforms(opt)
    test_data = FaceForensics(opt, split='test', transforms=all_transform)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=opt['datasets']['test']['batch_size'], shuffle=True,
        num_workers=opt['datasets']['n_workers'], sampler=None, drop_last=False)

    test_acc, test_loss, test_auc = validate(test_loader, model, criterion, args, opt)

@torch.no_grad()
def validate(val_loader, model, criterion, args, opt):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')

    progress = ProgressMeter(len(val_loader), [batch_time, losses, acc], prefix="Epoch: [{}]".format(0))

    end = time.time()
    y_trues = []
    y_preds = []

    save_dir = './saved_test_images'  
    os.makedirs(save_dir, exist_ok=True) 
    image_counter = 0  

    for i, (images, labels) in enumerate(val_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)

        if i < 10:
            for j in range(images.size(0)):
                vutils.save_image(images[j], f'{save_dir}/test_image_{image_counter}.png')  
                image_counter += 1

        preds = model(images)
        loss = criterion(preds, labels)

        acc1 = accuracy(preds, labels, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        acc.update(acc1, images.size(0))

        y_trues.extend(labels.cpu().numpy())
        prob = 1 - torch.softmax(preds, dim=1)[:, 0].cpu().numpy()
        y_preds.extend(prob)

        batch_time.update(time.time() - end)
        end = time.time()

        if (args.gpu is not None or args.local_rank == 0) and i % 20 == 0:
            progress.display(i)

    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    print(f' * Acc@1 {acc.avg:.3f}, auc {auc:.3f}')

    return acc.avg, losses.avg, auc

if __name__ == '__main__':
    main()
