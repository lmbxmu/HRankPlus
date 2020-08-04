
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
from torchvision import datasets, transforms

from models.cifar10.vgg import vgg_16_bn
from models.cifar10.resnet import resnet_56,resnet_110
from models.cifar10.googlenet import googlenet
from models.cifar10.densenet import densenet_40
from models.imagenet.resnet import resnet_50
from models.imagenet.mobilenetv2 import mobilenet_v2
from models.imagenet.mobilenetv1 import mobilenet_v1

from data import imagenet_dali
import utils.common as utils

parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10','imagenet'),
    help='dataset')
parser.add_argument(
    '--job_dir',
    type=str,
    default='result/tmp',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet','mobilenet_v2','mobilenet_v1'),
    help='The architecture to prune')
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default=None,
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--limit',
    type=int,
    default=5,
    help='The num of batch to get rank.')
parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
print('==> Preparing data..')
if args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

elif args.dataset=='imagenet':
    print('==> Preparing data..')
    def get_data_set():
        return imagenet_dali.get_imagenet_iter_dali('train', args.data_dir, args.batch_size,
                                                        num_threads=4, crop=224, device_id=0, num_gpus=1)
    train_loader = get_data_set()

# Model
print('==> Building model..')
net = eval(args.arch)(compress_rate=[0.]*100)
net = net.to(device)
print(net)

if len(args.gpu)>1 and torch.cuda.is_available():
    device_id = []
    for i in range((len(args.gpu) + 1) // 2):
        device_id.append(i)
    net = torch.nn.DataParallel(net, device_ids=device_id)

if args.pretrain_dir:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    if args.arch=='vgg_16_bn' or args.arch=='resnet_56':
        checkpoint = torch.load(args.pretrain_dir, map_location='cuda:'+args.gpu)
    else:
        checkpoint = torch.load(args.pretrain_dir)
    if args.arch=='mobilenet_v2' or args.arch=='resnet_50':
        net.load_state_dict(checkpoint)
    elif args.arch=='densenet_40':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(checkpoint['state_dict'])

criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0.)

#get feature map of certain layer via hook
def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total

def get_feature_hook_densenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total

def get_feature_hook_googlenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def inference():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    limit = args.limit

    with torch.no_grad():
        #for batch_idx, (inputs, targets) in enumerate(train_loader):
        for batch_idx, batch_data in enumerate(train_loader):
            #use the first 5 batches to estimate the rank.
            if batch_idx >= limit:
               break

            images = batch_data[0]['data'].cuda()
            targets = batch_data[0]['label'].squeeze().long().cuda()
            #inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(images)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            utils.progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))#'''


if args.arch=='vgg_16_bn':

    if len(args.gpu) > 1:
        relucfg = net.module.relucfg
    else:
        relucfg = net.relucfg

    for i, cov_id in enumerate(relucfg):
        cov_layer = net.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        if not os.path.isdir('rank_conv/'+args.arch+'_limit%d'%(args.limit)):
            os.mkdir('rank_conv/'+args.arch+'_limit%d'%(args.limit))
        np.save('rank_conv/'+args.arch+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())

        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='resnet_56':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit)+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet56 per block
    cnt=1
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(9):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='densenet_40':

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))

    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # Densenet per block & transition
    for i in range(3):
        dense = eval('net.dense%d' % (i + 1))
        for j in range(12):
            cov_layer = dense[j].relu
            if j==0:
                handler = cov_layer.register_forward_hook(get_feature_hook)
            else:
                handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
            inference()
            handler.remove()

            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/rank_conv%d'%(13*i+j+1)+'.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

        if i<2:
            trans=eval('net.trans%d' % (i + 1))
            cov_layer = trans.relu
    
            handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
            inference()
            handler.remove()

            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/rank_conv%d' % (13 * (i+1)) + '.npy', feature_result.numpy())
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)#'''

    cov_layer = net.relu
    handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
    inference()
    handler.remove()
    np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit) + '/rank_conv%d' % (39) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

elif args.arch=='googlenet':

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cov_list=['pre_layers',
              'inception_a3',
              'maxpool1',
              'inception_a4',
              'inception_b4',
              'inception_c4',
              'inception_d4',
              'maxpool2',
              'inception_a5',
              'inception_b5',
              ]

    # branch type
    tp_list=['n1x1','n3x3','n5x5','pool_planes']
    for idx, cov in enumerate(cov_list):

        cov_layer=eval('net.'+cov)

        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        if idx>0:
            for idx1,tp in enumerate(tp_list):
                if idx1==3:
                    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d_'%(idx+1)+tp+'.npy',
                            feature_result[sum(net.filters[idx-1][:-1]) : sum(net.filters[idx-1][:])].numpy())
                #elif idx1==0:
                #    np.save('rank_conv1/' + args.arch + '/rank_conv%d_'%(idx+1)+tp+'.npy',
                #            feature_result[0 : sum(net.filters[idx-1][:1])].numpy())
                else:
                    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d_' % (idx + 1) + tp + '.npy',
                            feature_result[sum(net.filters[idx-1][:idx1]) : sum(net.filters[idx-1][:idx1+1])].numpy())
        else:
            np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d' % (idx + 1) + '.npy',feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

elif args.arch=='resnet_110':

    cov_layer = eval('net.relu')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt = 1
    # ResNet110 per block
    for i in range(3):
        block = eval('net.layer%d' % (i + 1))
        for j in range(18):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch  + '_limit%d' % (args.limit) + '/rank_conv%d' % (
            cnt + 1) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch  + '_limit%d' % (args.limit) + '/rank_conv%d' % (
                cnt + 1) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='resnet_50':

    cov_layer = eval('net.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet50 per bottleneck
    cnt=1
    for i in range(4):
        block = eval('net.layer%d' % (i + 1))
        for j in range(net.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit) + '/rank_conv%d'%(cnt+1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            if j==0:
                np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                        feature_result.numpy())#shortcut conv
                cnt += 1
            np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (cnt + 1) + '.npy',
                    feature_result.numpy())#conv3
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='mobilenet_v2':
    cov_layer = eval('net.features[0]')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit)+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt=1
    for i in range(1,19):

        if i==1:
            block = eval('net.features[%d].conv' % (i))
            relu_list=[2,4]
        elif i==18:
            block = eval('net.features[%d]' % (i))
            relu_list=[2]
        else:
            block = eval('net.features[%d].conv' % (i))
            relu_list = [2,5,7]

        for j in relu_list:
            cov_layer = block[j]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

elif args.arch=='mobilenet_v1':

    cov_layer = eval('net.conv1[2]')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    if not os.path.isdir('rank_conv/' + args.arch + '_limit%d' % (args.limit)):
        os.mkdir('rank_conv/' + args.arch + '_limit%d' % (args.limit))
    np.save('rank_conv/' + args.arch + '_limit%d' % (args.limit) + '/rank_conv%d' % (1) + '.npy',
            feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    cnt=1
    for i in range(13):
        block = eval('net.features[%d]' % (i))
        relu_list = [2, 5]

        for j in relu_list:
            cov_layer = block[j]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
