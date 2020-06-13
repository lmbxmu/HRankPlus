import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, compress_rate, num_classes=1000):
        super(MobileNetV1, self).__init__()
        self.compress_rate=compress_rate
        #self.num_classes=num_classes

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride, inp_cprate, oup_cprate):
            return nn.Sequential(
                nn.Conv2d(int(inp*(1-inp_cprate)), int(inp*(1-inp_cprate)), 3, stride, 1, groups=int(inp*(1-inp_cprate)), bias=False),
                nn.BatchNorm2d(int(inp*(1-inp_cprate))),
                nn.ReLU(inplace=True),

                nn.Conv2d(int(inp*(1-inp_cprate)), int(oup*(1-oup_cprate)), 1, 1, 0, bias=False),
                nn.BatchNorm2d(int(oup*(1-oup_cprate))),
                nn.ReLU(inplace=True),
            )

        self.conv1 = conv_bn(3, 32, 2)

        self.features = nn.Sequential(
            conv_dw(32, 64, 1, 0., compress_rate[1]),
            conv_dw(64, 128, 2, compress_rate[1], compress_rate[2]),
            conv_dw(128, 128, 1, compress_rate[2], compress_rate[3]),
            conv_dw(128, 256, 2, compress_rate[3], compress_rate[4]),
            conv_dw(256, 256, 1, compress_rate[4], compress_rate[5]),
            conv_dw(256, 512, 2, compress_rate[5], compress_rate[6]),
            conv_dw(512, 512, 1, compress_rate[6], compress_rate[7]),
            conv_dw(512, 512, 1, compress_rate[7], compress_rate[8]),
            conv_dw(512, 512, 1, compress_rate[8], compress_rate[9]),
            conv_dw(512, 512, 1, compress_rate[9], compress_rate[10]),
            conv_dw(512, 512, 1, compress_rate[10], compress_rate[11]),
            conv_dw(512, 1024, 2, compress_rate[11], compress_rate[12]),
            conv_dw(1024, 1024, 1, compress_rate[12], 0.),
        )
        #self.classifier = nn.Linear(1024, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

def mobilenet_v1(compress_rate):
    return MobileNetV1(compress_rate=compress_rate)