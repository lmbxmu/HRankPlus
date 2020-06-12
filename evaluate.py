
import os
import numpy as np
import time, datetime
import torch
import random
import argparse
import pdb
from thop import profile

import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from torchvision import datasets, transforms

from models.imagenet.resnet import resnet_50
from models.imagenet.mobilenetv2 import mobilenet_v2
from models.imagenet.mobilenetv1 import mobilenet_v1

import utils.common as utils

parser = argparse.ArgumentParser("ImageNet training")

parser.add_argument(
    '--data_dir',
    type=str,
    default='',
    help='path to dataset')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_56',
    help='architecture')

parser.add_argument(
    '--job_dir',
    type=str,
    default='./models',
    help='path for saving trained models')


parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='batch size')

parser.add_argument(
    '--epochs',
    type=int,
    default=100,
    help='num of training epochs')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=0.1,
    help='init learning rate')

parser.add_argument(
    '--lr_decay_step',
    default='30,60',
    type=str,
    help='learning rate')

parser.add_argument(
    '--momentum',
    type=float,
    default=0.9,
    help='momentum')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='weight decay')

parser.add_argument(
    '--label_smooth',
    type=float,
    default=0.1,
    help='label smoothing')


parser.add_argument(
    '--use_pretrain',
    action='store_true',
    help='adjust ckpt from pruned checkpoint')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='',
    help='pretrain model path')

parser.add_argument(
    '--random_rule',
    type=str,
    default='hrank_pretrain',
    help='pretrain rule')

parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')

parser.add_argument(
    '-j', '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 4)')

args = parser.parse_args()

CLASSES = 1000
print_freq = (64*50)//args.batch_size

if not os.path.isdir(args.job_dir):
    os.mkdir(args.job_dir)

utils.record_config(args)
logger = utils.get_logger(os.path.join(args.job_dir, 'logger.log'))


def load_resnet_model(model, oristate_dict, random_rule):
    cfg = {'resnet18': [2, 2, 2, 2],
           'resnet34': [3, 4, 6, 3],
           'resnet_50': [3, 4, 6, 3],
           'resnet101': [3, 4, 23, 3],
           'resnet152': [3, 8, 36, 3]}

    state_dict = model.state_dict()

    current_cfg = cfg[args.arch]
    last_select_index = None

    all_honey_conv_weight = []

    prefix = "/home/zyc/HRank_Plus/rank_conv/resnet_50/rank_conv"
    subfix = ".npy"
    cnt=1

    conv_weight_name = 'conv1.weight'
    #logger.info(conv_weight_name)
    all_honey_conv_weight.append(conv_weight_name)
    oriweight = oristate_dict[conv_weight_name]
    curweight = state_dict[conv_weight_name]
    orifilter_num = oriweight.size(0)
    currentfilter_num = curweight.size(0)

    if orifilter_num != currentfilter_num and (
                    random_rule == 'random_pretrain' or random_rule == 'hrank_pretrain'):

        select_num = currentfilter_num
        if random_rule == 'random_pretrain':
            select_index = random.sample(range(0, orifilter_num - 1), select_num)
            select_index.sort()
        else:
            logger.info('loading rank from: ' + prefix + str(cnt) + subfix)
            rank = np.load(prefix + str(cnt) + subfix)
            select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
            select_index.sort()

        for index_i, i in enumerate(select_index):
                state_dict[conv_weight_name][index_i] = \
                    oristate_dict[conv_weight_name][i]

        last_select_index = select_index

    cnt+=1

    for layer, num in enumerate(current_cfg):
        layer_name = 'layer' + str(layer + 1) + '.'

        for k in range(num):
            if args.arch == 'resnet_18' or args.arch == 'resnet_34':
                iter = 2  # the number of convolution layers in a block, except for shortcut
            else:
                iter = 3
            if k==0:
                iter +=1
            for l in range(iter):
                record_last=True
                if k==0 and l==2:
                    conv_name = layer_name + str(k) + '.downsample.0'
                    record_last=False
                elif k==0 and l==3:
                    conv_name = layer_name + str(k) + '.conv' + str(l)
                else:
                    conv_name = layer_name + str(k) + '.conv' + str(l + 1)
                conv_weight_name = conv_name + '.weight'
                #logger.info(conv_weight_name)
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'hrank_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        logger.info('loading rank from: ' + prefix + str(cnt) + subfix)
                        rank = np.load(prefix + str(cnt) + subfix)
                        select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                        select_index.sort()

                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    if record_last:
                        last_select_index = select_index

                elif last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    if record_last:
                        last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    if record_last:
                        last_select_index = None

                cnt+=1


    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_mobilenetv2_model(model, oristate_dict, random_rule):

    state_dict = model.state_dict()

    last_select_index = None

    all_honey_conv_weight = []

    prefix = "/home/zyc/HRank_Plus/rank_conv/mobilenet_v2/rank_conv"
    subfix = ".npy"

    layer_cnt=1
    conv_cnt=1
    cfg=[1,2,3,4,3,3,1,1]
    #cfg_er=[1,6,6,6,6,6,6]
    for layer, num in enumerate(cfg):
        if layer_cnt==1:
            conv_id=[0,3]
        elif layer_cnt==18:
            conv_id=[0]
        else:
            conv_id=[0,3,6]

        for k in range(num):
            if layer_cnt==18:
                block_name = 'features.' + str(layer_cnt) + '.'
            else:
                block_name = 'features.'+str(layer_cnt)+'.conv.'

            for l in conv_id:
                conv_cnt += 1
                conv_name = block_name+str(l)
                conv_weight_name = conv_name + '.weight'
                all_honey_conv_weight.append(conv_weight_name)
                oriweight = oristate_dict[conv_weight_name]
                curweight = state_dict[conv_weight_name]
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                logger.info(conv_weight_name)

                if orifilter_num != currentfilter_num and (
                        random_rule == 'random_pretrain' or random_rule == 'hrank_pretrain'):

                    select_num = currentfilter_num
                    if random_rule == 'random_pretrain':
                        select_index = random.sample(range(0, orifilter_num - 1), select_num)
                        select_index.sort()
                    else:
                        logger.info('loading rank from: ' + prefix + str(conv_cnt) + subfix)
                        rank = np.load(prefix + str(conv_cnt) + subfix)
                        select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                        select_index.sort()

                    if (l==6 or (l==0 and layer_cnt!=1) or (l==3 and layer_cnt==1)) and last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[conv_weight_name][index_i][index_j] = \
                                    oristate_dict[conv_weight_name][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[conv_weight_name][index_i] = \
                                oristate_dict[conv_weight_name][i]

                    last_select_index = select_index

                elif  (l==6 or (l==0 and layer_cnt!=1) or (l==3 and layer_cnt==1)) and last_select_index is not None:
                    for index_i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][index_i][j]
                    last_select_index = None

                else:
                    state_dict[conv_weight_name] = oriweight
                    last_select_index = None

            layer_cnt+=1


    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)

def load_mobilenetv1_model(model, oristate_dict, random_rule):

    state_dict = model.state_dict()

    last_select_index = None

    all_honey_conv_weight = []

    prefix = "/home/zyc/HRank_Plus/rank_conv/mobilenet_v1/rank_conv"
    subfix = ".npy"

    conv_cnt=1
    for layer_cnt in range(13):
        conv_id=[0,3]

        block_name = 'features.'+str(layer_cnt)+'.'

        for l in conv_id:
            conv_cnt += 1
            conv_name = block_name+str(l)
            conv_weight_name = conv_name + '.weight'
            all_honey_conv_weight.append(conv_weight_name)
            oriweight = oristate_dict[conv_weight_name]
            curweight = state_dict[conv_weight_name]
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (
                    random_rule == 'random_pretrain' or random_rule == 'hrank_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num - 1), select_num)
                    select_index.sort()
                else:
                    logger.info('loading rank from: ' + prefix + str(conv_cnt) + subfix)
                    rank = np.load(prefix + str(conv_cnt) + subfix)
                    select_index = np.argsort(rank)[orifilter_num - currentfilter_num:]  # preserved filter id
                    select_index.sort()

                if l==3 and last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[conv_weight_name][index_i][index_j] = \
                                oristate_dict[conv_weight_name][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[conv_weight_name][index_i] = \
                            oristate_dict[conv_weight_name][i]

                last_select_index = select_index

            elif l==3 and last_select_index is not None:
                for index_i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[conv_weight_name][index_i][index_j] = \
                            oristate_dict[conv_weight_name][index_i][j]
                last_select_index = None

            else:
                state_dict[conv_weight_name] = oriweight
                last_select_index = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_name = name + '.weight'
            if conv_name not in all_honey_conv_weight:
                state_dict[conv_name] = oristate_dict[conv_name]

        elif isinstance(module, nn.Linear):
            state_dict[name + '.weight'] = oristate_dict[name + '.weight']
            state_dict[name + '.bias'] = oristate_dict[name + '.bias']

    model.load_state_dict(state_dict)


def main():

    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    if args.compress_rate:
        import re
        cprate_str = args.compress_rate
        cprate_str_list = cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                assert len(find_num) == 1
                num = int(find_num[0].replace('*', ''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        compress_rate = cprate

    # load model
    logger.info('compress_rate:' + str(compress_rate))
    logger.info('==> Building model..')
    model = eval(args.arch)(compress_rate=compress_rate).cuda()
    logger.info(model)
    #model = nn.DataParallel(model)

    # calculate model size
    input_image_size = 224
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    logger.info('Params: %.2f' % (params))
    logger.info('Flops: %.2f' % (flops))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = utils.CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    # split the weight parameter that need weight decay
    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if 'fc' in pname or 'conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    # define the optimizer
    optimizer = torch.optim.SGD(
        [{'params' : other_parameters},
        {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
        args.learning_rate,
        momentum=args.momentum,
        )

    # define the learning rate scheduler
    # we use the linear learning rate here
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//4, args.epochs//2, args.epochs//4*3], gamma=0.1)
    start_epoch = 0
    best_top1_acc= 0

    # load the checkpoint if it exists
    checkpoint_tar = os.path.join(args.job_dir, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        logger.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch']
        best_top1_acc = checkpoint['best_top1_acc']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("loaded checkpoint {} epoch = {}".format(checkpoint_tar, checkpoint['epoch']))  # '''
    else:
        if args.use_pretrain:
            logger.info('resuming from pretrain model')
            origin_model = eval(args.arch)(compress_rate=[0.] * 100).cuda()
            ckpt = torch.load(args.pretrain_dir)
            if args.arch=='mobilenet_v1':
                origin_model.load_state_dict(ckpt['state_dict'])
            else:
                origin_model.load_state_dict(ckpt)
            oristate_dict = origin_model.state_dict()
            if args.arch == 'resnet_50':
                load_resnet_model(model, oristate_dict, args.random_rule)
            elif args.arch == 'mobilenet_v2':
                load_mobilenetv2_model(model, oristate_dict, args.random_rule)
            elif args.arch == 'mobilenet_v1':
                load_mobilenetv1_model(model, oristate_dict, args.random_rule)
            else:
                raise
        else:
            logger.info('training from scratch')


    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    # load training data
    traindir = os.path.join(args.data_dir, 'ILSVRC2012_img_train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        utils.Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # train the model
    epoch = start_epoch
    while epoch < args.epochs:
        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model, criterion_smooth, optimizer, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.job_dir)

        epoch += 1

    training_time = (time.time() - start_t) / 36000
    logger.info('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model, criterion, optimizer, scheduler):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    '''progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))#'''

    model.train()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    logger.info('learning_rate: ' + str(cur_lr))

    num_iter = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits = model(images)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info(
                'Epoch[{0}]({1}/{2}): '
                'Loss {loss.avg:.4f} '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, i, num_iter, loss=losses,
                    top1=top1, top5=top5))
            #progress.display(i)

    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    '''progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')#'''

    # switch to evaluation mode
    num_iter = len(val_loader)
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = utils.accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logger.info(
                    'Epoch[{0}]({1}/{2}): '
                    'Loss {loss.avg:.4f} '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        epoch, i, num_iter, loss=losses,
                        top1=top1, top5=top5))
                #progress.display(i)

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
  main()
