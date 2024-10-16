import os
import timm.optim.optim_factory as optim_factory
import wandb
import argparse
import time
import random
import shutil
import warnings
import json
import yaml

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from torch.optim import lr_scheduler
from sklearn import metrics
from torch.autograd import Variable

# from datasets.ff_all import FaceForensics
from datasets.lamps import FaceForensics
from datasets.factory import create_data_transforms
# from model.LVNet import Two_Stream_Net
from model.LVNet_cls import Two_Stream_Net_Cls
from utils.utils import *
import torchvision.utils as vutils

torch.autograd.set_detect_anomaly(True)

def main():
    #### setup options of three networks
    parser = argparse.ArgumentParser()

    parser.add_argument("--opt", default='./config/FF++.yml', type=str, help="Path to option YMAL file.")
    
    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--device', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--mixup', default=False, help='using mixup augmentation.')
    
    parser.set_defaults(bottleneck=True)
    parser.set_defaults(verbose=True)

    args = parser.parse_args()

    opt = yaml.safe_load(open(args.opt, 'r'))
    seed = opt["train"]["manual_seed"]
        
    if seed is not None:
        if args.gpu is None:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
    else:  
        if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1

    ngpus_per_node = torch.cuda.device_count()

    main_worker(args.gpu, ngpus_per_node, args, opt)
        

def main_worker(gpu, ngpus_per_node, args, opt):

    args.gpu = gpu

    config = wandb.config
    config = vars(args)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(args.gpu)

    elif args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device(args.local_rank)

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    # create model
    print(f"Creating model: {opt['model']['baseline']}")
    model = Two_Stream_Net_Cls()
    model.to(args.device)

    if not opt['train']['resume'] == None:
        from_pretrained(model, opt)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                            output_device=args.local_rank, find_unused_parameters=True)
        param_groups = optim_factory.add_weight_decay(model.module, opt['train']['weight_decay'])
    else:
        param_groups = optim_factory.add_weight_decay(model, opt['train']['weight_decay'])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.Adam(param_groups, lr=opt['train']['lr'], betas=(0.9, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    cudnn.benchmark = True

    # Data loading code
    all_transform = create_data_transforms(opt)
    train_data = FaceForensics(opt, split='train', transforms=all_transform)
    val_data = FaceForensics(opt, split='val', transforms=all_transform)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, rank=args.local_rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, rank=args.local_rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt['datasets']['train']['batch_size'], shuffle=(train_sampler is None),
        num_workers=opt['datasets']['n_workers'], sampler=train_sampler, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=opt['datasets']['train']['batch_size'], shuffle=False,
        num_workers=opt['datasets']['n_workers'], sampler=val_sampler, drop_last=False)

    if (args.gpu is not None or args.local_rank == 0) and opt['train']['resume'] == None: 
        save_path = opt['train']['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        var_dict = vars(args)
        var_dict['optimizer'] = str(optimizer.__class__.__name__)
        var_dict['device'] = str(args.device)
        json_str = json.dumps(var_dict)
        with open(os.path.join(save_path, 'config.json'), 'w') as json_file:
            json_file.write(json_str)

    best = 0.0

    for epoch in range(opt['train']['start_epoch'], opt['train']['epoch']):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, args, opt)
        test_acc, test_loss, test_auc = validate(val_loader, model, criterion, epoch, args, opt)
        scheduler.step()

        is_best = test_auc > best
        best = max(test_auc, best)

        if args.gpu is not None or args.local_rank == 0:
            save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if args.gpu == None else model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, file=save_path, epoch=epoch)
        
            for param_group in optimizer.param_groups:
                cur_lr = param_group['lr']

            log_info = {
                "train_acc": train_acc,
                "train_loss": train_loss,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "test_auc": test_auc,
                "learning_rate": cur_lr
            }
            # wandb.log(log_info)

# Strange AUG Method...
def mixup_data(x, y, m, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    rd = np.random.rand(1)

    if rd > 0.5:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    m_a, m_b = m, m[index]

    return mixed_x, y_a, y_b, m_a, m_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_segLoss(segLoss, psegs, m_a, m_b, lam):
    return lam * segLoss(psegs, m_a) + (1 - lam) * segLoss(psegs, m_b)

def train(train_loader, model, criterion, optimizer, epoch, args, opt):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.5f')
    acc = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, acc], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    save_dir = './saved_train_images'
    os.makedirs(save_dir, exist_ok=True)
    image_counter = 0

    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.to(args.device)
        labels = labels.to(args.device)

        if i < 10:
            for j in range(images.size(0)):
                vutils.save_image(images[j], f'{save_dir}/train_image_{image_counter}.png')
                image_counter += 1

        # forward
        if args.mixup:
            inputs, labels_a, labels_b, lam = mixup_data(images, labels, device=args.device)
            inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))

            preds = model(inputs)

            # compute output
            loss = mixup_criterion(criterion, preds, labels_a, labels_b, lam)
            
        else:
            preds = model(images)
            loss = criterion(preds, labels)

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels, topk=(1,))[0]

        losses.update(loss.item(), images.size(0))
        acc.update(acc1, images.size(0))
        
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (args.gpu is not None or args.local_rank == 0) and i % 20 == 0:
            progress.display(i)

    return acc.avg, losses.avg

@torch.no_grad()
def validate(val_loader, model, criterion, epoch, args, opt):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.5f')
    acc = AverageMeter('Acc@1', ':6.2f')
    auc = AverageMeter('AUC', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, losses, acc, auc], prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    all_preds = []
    all_labels = []
    
    end = time.time()

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

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels, topk=(1,))[0]
        losses.update(loss.item(), images.size(0))
        acc.update(acc1, images.size(0))

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    #auc_score = metrics.roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds), multi_class="ovo")
    print(np.concatenate(all_labels).shape)
    print(np.concatenate(all_preds).shape)
    auc_score = metrics.roc_auc_score(np.concatenate(all_labels), np.concatenate(all_preds)[:, 1], multi_class="ovo")

    auc.update(auc_score, len(val_loader))

    if (args.gpu is not None or args.local_rank == 0):
        progress.display(len(val_loader))
        print(f' * Acc@1 {acc.avg:.3f} AUC {auc.avg:.3f}')

    return acc.avg, losses.avg, auc.avg

def save_checkpoint(state, is_best, epoch, file='checkpoint.pth.tar'):
    # if os.path.exists(filename):
    #     os.mkdir()
    filename = os.path.join(file, 'checkpoint-{:02d}.pth.tar'.format(epoch))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(file, 'model_best.pth.tar'))


def from_pretrained(model, opt):
    state_dict = torch.load(opt['train']['resume'], map_location='cpu')
    model.load_state_dict(cleanup_state_dict(state_dict['state_dict']), strict=False)

    opt['train']['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
    opt['train']['start_epoch'] = state_dict['epoch']


if __name__ == "__main__":
    main()
