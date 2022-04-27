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
from mit_semseg.dataset import TrainDataset
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
def train(segmentation_module, iterator, optimizers, history, epoch, cfg):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss_seg = AverageMeter()
    ave_total_loss_disc = AverageMeter()
    ave_acc = AverageMeter()
    
    segmentation_module.train(not cfg.TRAIN.fix_bn)
    # freeze all weights besides the last layer
    for param in segmentation_module.parameters():
        param.requires_grad = False
    list(segmentation_module.parameters())[-1].requires_grad = True
    list(segmentation_module.parameters())[-2].requires_grad = True
    list(segmentation_module.parameters())[-3].requires_grad = True

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        start_t = time.time()
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()
        # print("read data :", time.time()-start_t)

        # adjust learning rate
        start_t = time.time()
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)
        # print("adjust learning rate :", time.time()-start_t)

        # forward pass
        start_t = time.time()
        loss_seg, loss_disc, acc = segmentation_module(batch_data)
        loss_seg = loss_seg.mean()
        loss_disc = loss_disc.mean()
        loss = loss_seg + loss_disc
        acc = acc.mean()
        # print("forward :", time.time()-start_t)

        # Backward
        start_t = time.time()
        try:
            loss.backward()
        except:
            import ipdb;ipdb.set_trace()
        # print("loss :", time.time()-start_t)
        for optimizer in optimizers:
            start_t = time.time()
            optimizer.step()
            # print("opt :", time.time()-start_t)

        start_t = time.time()
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss_seg.update(loss_seg.data.item())
        ave_total_loss_disc.update(loss_disc.data.item())
        ave_acc.update(acc.data.item()*100)
        # print("updates :", time.time()-start_t)
        # calculate accuracy, and display
        start_t = time.time()
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss seg: {:.6f}, Loss disc: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss_seg.average(), ave_total_loss_disc.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())
        #print("display :", time.time()-start_t)

def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, _, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


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


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, discriminator, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    if discriminator != None:
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    else:
        optimizer_discriminator = None
    return (optimizer_encoder, optimizer_decoder, optimizer_discriminator)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder, _) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


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
    if torch.cuda.is_available():
        discriminator.cuda()
    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, discriminator, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, discriminator, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    if torch.cuda.is_available():
        for ele in loader_train:
            for keys in ele[0].keys():
                ele[0][keys].cuda()

    # import pickle
    # l_heights = []
    # l_widths = []
    # acc = 0
    # for ele in loader_train:
    #     acc +=1
    #     for keys in ele[0].keys():
    #         print(keys,"=", ele[0][keys].shape)
    #         a = input()
    #         try:
    #             _,_,h,w = ele[0][keys].shape
    #             l_heights.append(h)
    #             l_widths.append(w)
    #         except:
    #             _,h,w = ele[0][keys].shape
    #             l_heights.append(h)
    #             l_widths.append(w)
    #     if acc == 1000:
    #         with open('eda_sizes/heights.pkl', 'wb') as f:
    #             pickle.dump(l_heights, f)
    #         with open('eda_sizes/widths.pkl', 'wb') as f:
    #             pickle.dump(l_widths, f)
    #         import ipdb;ipdb.set_trace()

    # create loader iterator
    iterator_train = iter(loader_train)
    # load nets into gpu
    # if len(gpus) > 1:
    #     segmentation_module = UserScatteredDataParallel(
    #         segmentation_module,
    #         device_ids=gpus)
    #     # For sync bn
    #     patch_replication_callback(segmentation_module)
    if torch.cuda.is_available():
        segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, discriminator, crit)

    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg)

        # checkpointing
        start_time = time.time()
        checkpoint(nets, history, cfg, epoch+1)
        print("chkp :", time.time()-start_time)
    print('Training Done!')


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
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

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
