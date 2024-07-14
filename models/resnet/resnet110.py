import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import torch.nn.functional as F

class ResNet110(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet110, self).__init__()
        self.in_planes = 16

        # Initial convolution and batch norm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # ResNet-110 has 18 blocks per stage for 3 stages, with a total of 54 blocks
        self.layer1 = self._make_layer(resnet.BasicBlock, 16, 18)
        self.layer2 = self._make_layer(resnet.BasicBlock, 32, 18, stride=2)
        self.layer3 = self._make_layer(resnet.BasicBlock, 64, 18, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * resnet.BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
