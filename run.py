from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
np.random.seed(1)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

from utils.lr_schedulers import WarmupMultiStepLR
from utils.video_loader import VideoDataset
import utils.transforms as T
import models
from models.losses import CrossEntropyLabelSmooth, TripletLoss, RegularLoss
from utils.utils import AverageMeter, Logger, save_checkpoint
from utils.eval_metrics import evaluate
from utils.samplers import RandomIdentitySampler

from data import data_manager

def main(args=None):
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    # torch.cuda.set_device(0)
    use_gpu = torch.cuda.is_available()

    if args.use_cpu:use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, args.log_train))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, args.log_test))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    transform_train = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(),
    ])

    transform_train2 = T.Compose([
        T.Resize((args.height, args.width)),
        T.Random2DTranslation(args.height, args.width),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = False

    trainloader = DataLoader(
        VideoDataset(dataset.train, data_name=args.dataset, seq_len=args.seq_len, sample='random', transform=transform_train, transform2=transform_train2, type = "train"),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, data_name=args.dataset, seq_len=args.seq_len, sample='dense', transform=transform_test, type="test"),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, data_name=args.dataset, seq_len=args.seq_len, sample='dense', transform=transform_test, type="test"),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing models: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, final_dim = args.feat_dim)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    crossEntropyLoss = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    tripletLoss = TripletLoss(margin=args.margin)
    regularLoss = RegularLoss(use_gpu=use_gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = WarmupMultiStepLR(optimizer, args.stepsize, args.gamma, args.warmup_factor, args.warmup_items, args.warmup_method)
    start_epoch = args.start_epoch

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        atest(model, queryloader, galleryloader, use_gpu)
        return

    start_time = time.time()
    best_rank1 = -np.inf
    for epoch in range(start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))

        train(model, crossEntropyLoss, tripletLoss, regularLoss, optimizer, trainloader, use_gpu)

        # if args.stepsize > 0:
        scheduler.step()

        if (epoch+1) >= 200 and (epoch+1) % args.eval_step == 0:
            print("==> Test")
            rank1 = atest(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best: best_rank1 = rank1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
            }, is_best, osp.join(args.save_dir, args.model_name + str(epoch+1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train(model, crossEntropyLoss, tripletLoss, regularLoss, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    print('batch number:' + str(len(trainloader)))
    for batch_idx, (imgs, pids, _, head_map, body_map, leg_map) in enumerate(trainloader):
        # print(img_align)
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
        imgs, pids = Variable(imgs), Variable(pids)
        global_label, global_f, region1_label, region1_f, region2_label, region2_f, region3_label, region3_f, \
            align_output1, align_output2, align_output3, align_output4, align_output5, align_output6\
            = model(imgs, head_map, body_map, leg_map)
        loss = calculate_loss(global_label, global_f, region1_label, region1_f, region2_label, region2_f, region3_label, region3_f,\
                              align_output1, align_output2, align_output3, align_output4, align_output5, align_output6,\
                              crossEntropyLoss, tripletLoss, regularLoss, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), pids.size(0))

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss1 {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))

def calculate_loss(global_label, global_f, region1_label, region1_f, region2_label, region2_f, region3_label, region3_f,\
                    align_output1, align_output2, align_output3, align_output4, align_output5, align_output6,\
                    crossEntropyLoss, tripletLoss, regularLoss, pids):
    # global branch
    global_cross_loss = crossEntropyLoss(global_label, pids)
    global_trip_loss, _, _ = tripletLoss(global_f, pids)
    global_loss = global_cross_loss + global_trip_loss
    # align branch
    region1_cross_loss = crossEntropyLoss(region1_label, pids)
    region1_trip_loss, _, _ = tripletLoss(region1_f, pids)
    region2_cross_loss = crossEntropyLoss(region2_label, pids)
    region2_trip_loss, _, _ = tripletLoss(region2_f, pids)
    region3_cross_loss = crossEntropyLoss(region3_label, pids)
    region3_trip_loss, _, _ = tripletLoss(region3_f, pids)
    region_loss = region1_cross_loss + region1_trip_loss + region2_cross_loss + region2_trip_loss + region3_cross_loss + region3_trip_loss
    # consistency Regularization
    regular_loss1 = regularLoss(align_output1)
    regular_loss2 = regularLoss(align_output2)
    regular_loss3 = regularLoss(align_output3)
    regular_loss4 = regularLoss(align_output4)
    regular_loss5 = regularLoss(align_output5)
    regular_loss6 = regularLoss(align_output6)
    regular_loss = regular_loss1 + regular_loss2 + regular_loss3 + regular_loss4 + regular_loss5 + regular_loss6
    # print("global loss: " + str(global_loss.item()) + "; part loss: " + str(part_loss.item()) + "; region loss: " + str(region_loss.item()) + "; regular loss: " + str(regular_loss.item()))
    loss = args.a1 * global_loss + args.a2 * region_loss + args.a3 * regular_loss
    return loss

def atest(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    with torch.no_grad():
        model.eval()
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, head_map, body_map, leg_map) in enumerate(queryloader):
            if batch_idx % 100 == 0:
                print("current query:" + str(batch_idx))
            if use_gpu:
                imgs = imgs.cuda()
            b, n, s, c, h, w = imgs.size()
            assert(b==1)
            imgs = imgs.view(b*n, s, c, h, w)
            features = model(imgs, head_map, body_map, leg_map)
            features = features.view(n, -1)
            features = torch.mean(features, 0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)

        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        print('gallery num:' + str(len(galleryloader)))

        for batch_idx, (imgs, pids, camids, head_map, body_map, leg_map) in enumerate(galleryloader):
            if batch_idx % 100 == 0:
                print("current gallery:" + str(batch_idx))
            if use_gpu:
                imgs = imgs.cuda()
            # imgs = Variable(imgs, volatile=True)
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b*n, s , c, h, w)
            assert(b==1)
            features = model(imgs, head_map, body_map, leg_map)
            features = features.view(n, -1)
            features = torch.mean(features, 0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.numpy()

        print("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
        print("------------------")

    return cmc[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train video models with cross entropy loss')
    # Datasets
    parser.add_argument('-d', '--dataset', type=str, default='mars',
                        choices=data_manager.get_names())
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image (default: 224)")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image (default: 112)")
    parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
    # Optimization options
    parser.add_argument('--max-epoch', default=400, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--train-batch', default=32, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
    parser.add_argument('--lr', '--learning-rate', default=0.00035, type=float,
                        help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
    parser.add_argument('--stepsize', default=(100, 200, 300), type=tuple,
                        help="stepsize to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")
    parser.add_argument('--margin', type=float, default=0.5, help="margin for triplet loss")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="number of instances per identity")
    parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
    parser.add_argument('--seed', type=int, default=1, help="manual seed")
    parser.add_argument('--evaluate', default=False, action='store_true', help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=10,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--save-dir', type=str, default='log/final_log')
    parser.add_argument('--use-cpu', action='store_true', help="use cpu")
    parser.add_argument('--gpu-devices', default='4', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='Net', type=str, help='models name')
    parser.add_argument('--warmup_factor', default=0.01, type=float, help='warmup factor')
    parser.add_argument('--warmup_items', default=10, type=int, help='warmup items')
    parser.add_argument('--warmup_method', default='linear', type=str, help='warmup method')
    parser.add_argument('--log_train', default='log_train.txt', type=str, help='train log file name')
    parser.add_argument('--log_test', default='log_test.txt', type=str, help='test log file name')
    parser.add_argument('--model_name', default='checkpoint_ep', type=str, help='model file name')
    parser.add_argument('--feat_dim', default=1024, type=int, help='feature dim is feat_dim x 4')
    parser.add_argument('--a1', default=1, type=int, help='global loss weight')
    parser.add_argument('--a2', default=1, type=int, help='region loss weight')
    parser.add_argument('--a3', default=0.0003, type=int, help='regular loss weight')
    args = parser.parse_args()
    main(args)
