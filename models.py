import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):

    def __init__(self, input_dim, nr_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, nr_classes)

    def forward(self, x):
        return self.fc(x)


class FNNet(nn.Module):

    def __init__(self, input_dim, interm_dim, output_dim):
        super(FNNet, self).__init__()

        self.input_dim = input_dim
        self.dp1 = torch.nn.Dropout(0.2)
        self.dp2 = torch.nn.Dropout(0.2)
        self.fc1 = nn.Linear(input_dim, interm_dim)
        self.fc2 = nn.Linear(interm_dim, interm_dim)
        self.fc3 = nn.Linear(interm_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc3(x)
        return x

    def embed(self, x):
        x = self.dp1(F.relu(self.fc1(x.view(-1, self.input_dim))))
        x = self.dp2(F.relu(self.fc2(x)))
        return x


class ConvNet(nn.Module):
    def __init__(self, output_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.dp1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dp2 = torch.nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * 64, 128)
        self.dp3 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc2(x)
        return x

    def embed(self, x):
        x = F.relu(self.dp1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dp2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.dp3(self.fc1(x)))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.embed(x)
        out = self.linear(out)
        return out

    def embed(self, x):
        out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# ----------------------------------------------------------------------
# WideResNet (Zagoruyko and Komodakis, 2016), used by Sec. 5.2.3 (WRN-16-4,
# 2.7M params) and Sec. 5.5 (WRN-28-10). depth = 6n + 4, so n = (depth-4)/6
# residual blocks per one of 3 stages; channel widths [16, 16k, 32k, 64k]
# for widen_factor k. Standard pre-activation BN-ReLU-Conv block ordering,
# as in the paper and the reference torchvision/wide-resnet implementations.
# ----------------------------------------------------------------------
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """Wide Residual Network (Zagoruyko and Komodakis, 2016) for 32x32 inputs.

    Args:
        depth (int): total depth, ``6n + 4`` (paper uses 16 for Sec. 5.2.3,
            28 for Sec. 5.5).
        widen_factor (int): channel width multiplier ``k`` (paper: 4 for
            Sec. 5.2.3's WRN-16-4, 10 for Sec. 5.5's WRN-28-10).
        num_classes (int): output dimension.
        dropout_rate (float): dropout after the first conv of each block
            (Appendix C: "dropout with a rate of 0.4 for SVHN"; 0 elsewhere,
            and Sec. 5.5 states WRN-28-10 is used "without dropout").
        in_channels (int): input channels (3 for CIFAR-10/SVHN; also used
            for the 1-channel mel-spectrogram input of Sec. 5.5, replicated
            to 3 channels upstream, so this is left at 3 by default).
    """

    def __init__(self, depth=16, widen_factor=4, num_classes=10, dropout_rate=0.0, in_channels=3):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, 'WideResNet depth must be 6n + 4'
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(widths[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(widths[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(widths[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn_final = nn.BatchNorm2d(widths[3])
        self.embed_dim = widths[3]
        self.linear = nn.Linear(widths[3], num_classes)

    def _make_layer(self, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasicBlock(self.in_planes, planes, stride=s, dropout_rate=dropout_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.embed(x)
        return self.linear(out)

    def embed(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)


def WideResNet16_4(num_classes=10, dropout_rate=0.0, in_channels=3):
    """WRN-16-4, ~2.7M params -- Sec. 5.2.3's architecture."""
    return WideResNet(depth=16, widen_factor=4, num_classes=num_classes,
                      dropout_rate=dropout_rate, in_channels=in_channels)


def WideResNet28_10(num_classes=10, dropout_rate=0.0, in_channels=3):
    """WRN-28-10 -- Sec. 5.5's architecture ("without dropout")."""
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes,
                      dropout_rate=dropout_rate, in_channels=in_channels)


# ----------------------------------------------------------------------
# VGG16 (Simonyan and Zisserman, 2015), adapted for 32x32 inputs -- Table 1 /
# Table 2's transfer-target architecture. Standard "VGG for CIFAR" adaptation
# (5 max-pool halvings bring 32x32 down to 1x1, so no other change is needed
# versus the ImageNet architecture besides the classifier head).
# ----------------------------------------------------------------------
_VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG16(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(VGG16, self).__init__()
        layers = []
        c = in_channels
        for v in _VGG16_CFG:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(c, v, kernel_size=3, padding=1, bias=False),
                          nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.features = nn.Sequential(*layers)
        self.embed_dim = 512
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.embed(x)
        return self.classifier(out)

    def embed(self, x):
        out = self.features(x)
        return out.view(out.size(0), -1)


# ----------------------------------------------------------------------
# MobileNetV2 (Sandler et al., 2018), adapted for 32x32 inputs -- the first
# conv and the first inverted-residual block use stride 1 instead of 2
# (the standard "MobileNetV2 for CIFAR" adaptation the paper refers to as
# "kernel strides and pooling kernel sizes reduced to accommodate 32x32
# images"), so the network still downsamples to a 1x1 feature map by the end.
# ----------------------------------------------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                               groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.use_residual = stride == 1 and in_planes == out_planes

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.use_residual:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    _CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=10, in_channels=3):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layers = []
        in_planes = 32
        for expansion, out_planes, num_blocks, stride in self._CFG:
            for i in range(num_blocks):
                s = stride if i == 0 else 1
                layers.append(InvertedResidual(in_planes, out_planes, expansion, s))
                in_planes = out_planes
        self.layers = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(in_planes, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.embed_dim = 1280
        self.linear = nn.Linear(1280, num_classes)

    def forward(self, x):
        out = self.embed(x)
        return self.linear(out)

    def embed(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu6(self.bn2(self.conv2(out)))
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)
