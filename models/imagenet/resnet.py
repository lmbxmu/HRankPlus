
import torch.nn as nn

stage_repeat = [3, 4, 6, 3]
stage_out_channel = [64] + [256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3


def adapt_channel(compress_rate):

    stage_oup_cprate = []
    stage_oup_cprate += [compress_rate[0]]
    for i in range(len(stage_repeat)-1):
        stage_oup_cprate += [compress_rate[i+1]] * stage_repeat[i]
    stage_oup_cprate +=[0.] * stage_repeat[-1]

    mid_scale_cprate = compress_rate[len(stage_repeat):]

    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
        else:
            overall_channel += [int(stage_out_channel[i] * (1-stage_oup_cprate[i]))]
            mid_channel += [int(stage_out_channel[i]//4 * (1-mid_scale_cprate[i-1]))]

    return overall_channel, mid_channel


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    def __init__(self, midplanes, inplanes, planes, stride=1, is_downsample=False):
        super(Bottleneck, self).__init__()
        expansion = 4

        #midplanes = int(planes/expansion)
        norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, midplanes)
        self.bn1 = norm_layer(midplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(midplanes, midplanes, stride)
        self.bn2 = norm_layer(midplanes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv1x1(midplanes, planes)
        self.bn3 = norm_layer(planes)
        self.relu3 = nn.ReLU(inplace=True)

        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes
        self.midplanes = midplanes

        self.is_downsample = is_downsample
        self.expansion = expansion

        if is_downsample:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                norm_layer(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, compress_rate, num_classes=1000):
        super(ResNet50, self).__init__()

        overall_channel, mid_channel = adapt_channel(compress_rate)
        self.num_blocks = stage_repeat

        layer_num =0
        self.conv1 = nn.Conv2d(3, overall_channel[layer_num], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(overall_channel[layer_num])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.ModuleList()
        self.layer2 = nn.ModuleList()
        self.layer3 = nn.ModuleList()
        self.layer4 = nn.ModuleList()

        layer_num += 1
        for i in range(len(stage_repeat)):
            if i == 0:
                eval('self.layer%d' % (i+1)).append(Bottleneck(mid_channel[layer_num-1], overall_channel[layer_num-1], overall_channel[layer_num], stride=1, is_downsample=True))
                layer_num += 1
            else:
                eval('self.layer%d' % (i+1)).append(Bottleneck(mid_channel[layer_num-1], overall_channel[layer_num-1], overall_channel[layer_num], stride=2, is_downsample=True))
                layer_num += 1

            for j in range(1, stage_repeat[i]):
                eval('self.layer%d' % (i+1)).append(Bottleneck(mid_channel[layer_num-1], overall_channel[layer_num-1], overall_channel[layer_num]))
                layer_num +=1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)
        for i, block in enumerate(self.layer4):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_50(compress_rate):
    return ResNet50(compress_rate=compress_rate)