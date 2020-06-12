import torch
import torch.nn as nn
import torch.nn.functional as F

import math

norm_mean, norm_var = 0.0, 1.0

cov_cfg=[(3*i+1) for i in range(12*3+2+1)]


class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, dropRate=0):
        super(DenseBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                               padding=1, bias=False)

        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, compress_rate, depth=40, block=DenseBasicBlock,
        dropRate=0, num_classes=10, growthRate=12, compressionRate=1):
        super(DenseNet, self).__init__()
        self.compress_rate=compress_rate

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6

        transition = Transition

        self.covcfg=cov_cfg

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)

        self.dense1 = self._make_denseblock(block, n, compress_rate[1:n+1])
        self.trans1 = self._make_transition(transition, compressionRate, compress_rate[n+1])
        self.dense2 = self._make_denseblock(block, n, compress_rate[n+2:2*n+2])
        self.trans2 = self._make_transition(transition, compressionRate, compress_rate[2*n+2])
        self.dense3 = self._make_denseblock(block, n, compress_rate[2*n+3:3*n+3])
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, compress_rate):
        layers = []
        for i in range(blocks):
            layers.append(block(self.inplanes, outplanes=int(self.growthRate*(1-compress_rate[i])), dropRate=self.dropRate))
            self.inplanes += int(self.growthRate*(1-compress_rate[i]))

        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate, compress_rate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes*(1-compress_rate) // compressionRate))
        self.inplanes = outplanes
        return transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def densenet_40(compress_rate):
    return DenseNet(compress_rate=compress_rate, depth=40, block=DenseBasicBlock)
