import torch
import torch.nn as nn

class bottleBlock(nn.Module):
    expansion = 4
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(bottleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch * self.expansion, 3),
            nn.BatchNorm2d(out_ch * self.expansion),
        )
    def forward(self, x):
        identity = x
        x = self.conv(x)
        x += identity
        x = self.relu(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
    def forward(self, x):
        x = self.up(x)
        return x


class MyUNet(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self,in_ch, out_ch, n_classes):
        super(MyUNet, self).__init__()
        self.conv1 = bottleBlock(in_ch, 64)
        self.conv2 = bottleBlock(64, 128)
        self.conv3 = bottleBlock(128, 256)
        self.conv4 = bottleBlock(256, 512)
        self.conv5 = bottleBlock(512, 1024)
        self.conv6 = bottleBlock(1024, 512)
        self.conv7 = bottleBlock(512, 256)
        self.conv8 = bottleBlock(256, 128)
        self.conv9 = bottleBlock(128, 64)
        self.conv10 = bottleBlock(64, out_ch,1)

        self.mp = nn.MaxPool2d(2)
        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)

    def forward(self, x):
        x1 = self.mp(self.conv0(x))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        x5 = self.mp(self.conv4(x4))

        up_1 = self.up1(x5)
        merge1 = torch.cat([up_1, x4], dim=1)
        x6 = self.conv6(merge1)
        up_2 = self.up2(x6)

        merge2 = torch.cat([up_2, x3], dim=1)
        x7 = self.conv7(merge2)
        up_3 = self.up3(x7)

        merge3 = torch.cat([up_3, x2], dim=1)
        x8 = self.conv8(merge3)
        up_4 = self.up4(x8)

        merge4 = torch.cat([up_4, x1], dim=1)
        x9 = self.conv9(merge4)
        x10 = self.conv10(x9)
        return x10