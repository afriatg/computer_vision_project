# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset, ValDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback

import torch.nn.functional as F
import time

class Discriminator(nn.Module):
    def __init__(self):
      super(Discriminator, self).__init__()
      self.conv1 = nn.Conv2d(154, 32, 3, 3)
      self.conv2 = nn.Conv2d(32, 16, 3, 3)
      self.dropout1 = nn.Dropout(0.25)
      self.dropout2 = nn.Dropout(0.5)
      self.fc1 = nn.Linear(5040, 256)
      self.fc2 = nn.Linear(256, 32)
      self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
      x = self.conv1(x)
      x = F.relu(x)
      x = self.conv2(x)
      x = F.relu(x)
      x = self.dropout1(x)
      x = torch.flatten(x, 1)
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      output = F.softmax(x, dim=1)
      return output

# train one epoch
def test2(segmentation_module, loader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss_seg = AverageMeter()
    ave_total_loss_disc = AverageMeter()
    segmentation_module.eval()
    ave_acc = AverageMeter()
    
    # main loop
    tic = time.time()
    num_iter = 0
    for batch_data in loader:
        num_iter +=1
        # load a batch of data
        start_t = time.time()
        data_time.update(time.time() - tic)

        # forward pass
        start_t = time.time()
        loss_seg, loss_disc, acc = segmentation_module(batch_data, use_disc)
        loss_seg = loss_seg.mean()
        if use_disc:
            loss_disc = loss_disc.mean()
        acc = acc.mean()
        # print("forward :", time.time()-start_t)

        # Backward
        start_t = time.time()
        # print("loss :", time.time()-start_t)

        start_t = time.time()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss_seg.update(loss_seg.data.item())
        if use_disc:
            ave_total_loss_disc.update(loss_disc.data.item())
        else:
            ave_total_loss_disc.update(0)
        ave_acc.update(acc.data.item()*100)
        # print("updates :", time.time()-start_t)
        # calculate accuracy, and display
        start_t = time.time()
        print('Img [{}/{}], Time: {:.2f}, Data: {:.2f}, '
                'Accuracy: {:4.2f}, Loss seg: {:.6f}, Loss disc: {:.6f}'
                .format(num_iter, len(loader),
                        batch_time.average(), data_time.average(),
                        ave_acc.average(), ave_total_loss_seg.average(), ave_total_loss_disc.average()))

        #print("display :", time.time()-start_t)

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def main(cfg, gpus):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)
    crit = nn.NLLLoss(ignore_index=-1)
    discriminator = Discriminator()

    if args.save_disc!=None:
        discriminator.load_state_dict(torch.load(args.save_disc, map_location=lambda storage, loc: storage), strict=False)

    if torch.cuda.is_available():
        discriminator.cuda()
    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, discriminator, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, discriminator, crit)

    # Dataset and Loader
    dataset_train = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET)

    loader_test = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)

    if torch.cuda.is_available():
        for ele in loader_test:
            for keys in ele[0].keys():
                ele[0][keys].cuda()

    if torch.cuda.is_available():
        segmentation_module.cuda()

    test2(segmentation_module, loader_test)
    print('Test Done!')


if __name__ == '__main__':

    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        #default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        default = "config/ade20k-resnet50-upernet.yaml",
        #default = "config/ade20k-resnet18dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "--save_disc",
        default=None,
        help="discriminator_epoch_1.pth"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use_disc",
        default=0,
        type = int,
    )
    args = parser.parse_args()
    print(args, flush=True)
    use_disc = args.use_disc
    
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    if use_disc:
        cfg.DIR = cfg.DIR+'-GAN'
    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)
    main(cfg, gpus)
