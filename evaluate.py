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

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = False

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

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        atest(model, queryloader, galleryloader, use_gpu)
        return

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
    parser.add_argument('--evaluate', default=True, action='store_true', help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=10,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--save-dir', type=str, default='log')
    parser.add_argument('--use-cpu', action='store_true', help="use cpu")
    parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--arch', default='Net', type=str, help='models name')
    parser.add_argument('--log_train', default='test.txt', type=str, help='train log file name')
    parser.add_argument('--log_test', default='test.txt', type=str, help='test log file name')
    parser.add_argument('--model_path', default='./weights/rapa_model.pth.tar', type=str, help='model file name')
    parser.add_argument('--feat_dim', default=1024, type=int, help='feature dim is feat_dim x 4')
    args = parser.parse_args()
    main(args)