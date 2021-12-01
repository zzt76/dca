import torch
import torch.nn as nn
import torch.nn.functional as F
import geffnet
from torchinfo import summary

EPS = 1e-7


class DCA(nn.Module):
    def __init__(self, cout=1):
        super(DCA, self).__init__()
        print('Building Encoder, Decoder ...')
        self.encoder = Encoder()
        self.decoder = Decoder(cout=cout)
        print('Finished building model')

    def forward(self, x):
        parallel, features = self.encoder(x)
        out = self.decoder(x, features, parallel)
        return out


class Encoder(nn.Module):
    def __init__(self, basemodel_name='tf_efficientnet_b5_ap', pretrained=True):
        super(Encoder, self).__init__()
        print(f'Loading base model {basemodel_name} ...')
        torch.hub.set_dir('./.cache')
        self.basemodel = getattr(geffnet, basemodel_name)(pretrained=pretrained)
        # basemodel = geffnet.tf_efficientnet_b5_ap(pretrained=pretrained)
        self.basemodel.global_pool = nn.Identity()
        self.basemodel.classifier = nn.Identity()
        # self.un_flatten = nn.Unflatten(1, (2048, 8, 10))
        print(f'Finished loading base model {basemodel_name}')
        parallel = 32
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(3, parallel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(parallel),
            nn.GELU(),
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(parallel, parallel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(parallel),
            nn.GELU(),
        )
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(parallel, 48, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
        )

    def forward(self, x):
        parallel = self.conv_p1(x)
        parallel = self.conv_p2(parallel)
        residual = self.conv_d1(parallel)
        residual = self.conv_d2(residual)  # connect this after act1
        features = [x]

        for k, v in self.basemodel._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                if k == 'act1':
                    fuse = self.conv_d3(torch.cat([features[-1], residual], dim=1))
                    features.append(v(fuse))
                else:
                    features.append(v(features[-1]))
        return parallel, features


class Decoder(nn.Module):
    def __init__(self, cout, features=2048):
        super(Decoder, self).__init__()
        self.att_proc1 = AttentionProc(24)
        self.att_proc2 = AttentionProc(40)
        self.att_proc3 = AttentionProc(64)
        self.att_proc4 = AttentionProc(176)

        self.att_block1 = AttentionBlock(3, 24, 24)
        self.att_block2 = AttentionBlock(24, 40, 40)
        self.att_block3 = AttentionBlock(40, 64, 64)
        self.att_block4 = AttentionBlock(64, 176, 176)

        self.up_block0 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0)
        self.up_block1 = DecoderBlock(features, 176, features//2)
        self.up_block2 = DecoderBlock(features//2, 64, features//4)
        self.up_block3 = DecoderBlock(features//4, 40, features//8)
        self.up_block4 = DecoderBlock(features//8, 24, features//16)
        self.up_block5 = DecoderBlock(features//16, 32, features//16)
        self.out_block = nn.Sequential(
            DepthwiseDilatedConv(features//16, features//16),
            nn.Conv2d(features//16, features//16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features//16),
            nn.GELU(),
            nn.Conv2d(features//16, cout, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, features, parallel):
        feature1, feature2, feature3, feature4, feature5 = [features[4], features[5],
                                                            features[6], features[8], features[13]]

        att1 = self.att_block1(x, self.att_proc1(feature1))  # [24, 240, 320], [24, 120, 160]
        att2 = self.att_block2(att1, self.att_proc2(feature2))  # [40, 120, 160], [40, 60, 80]
        att3 = self.att_block3(att2, self.att_proc3(feature3))  # [64, 60, 80], [64, 30, 40]
        att4 = self.att_block4(att3, self.att_proc4(feature4))  # [176, 30, 40]

        up0 = self.up_block0(feature5)  # [2048, 15, 20]
        up1 = self.up_block1(up0, feature4, att4)  # [1024, 30, 40]
        up2 = self.up_block2(up1, feature3, att3)  # [512, 60, 80]
        up3 = self.up_block3(up2, feature2, att2)  # [256, 120, 160]
        up4 = self.up_block4(up3, feature1, att1)  # [128, 240, 320]
        up5 = self.up_block5(up4, parallel)
        out = self.out_block(up5)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels, trans_kernel_size=4):
        super(DecoderBlock, self).__init__()
        # dilation=1,3,5,7 of encoded feature
        self.dilation_conv = DepthwiseDilatedConv(feature_channels, feature_channels)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels+feature_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x, feature, atts=None):
        # upsample coming up feature to match encoded feature
        up = F.interpolate(x, size=feature.shape[2:], mode='bilinear', align_corners=True)
        dils = self.dilation_conv(feature)
        if atts is not None:
            c1 = self.conv1(torch.cat([up, dils*atts], dim=1))
        else:
            c1 = self.conv1(torch.cat([up, dils], dim=1))
        c2 = self.conv2(c1)
        return c2


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, feature_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels+feature_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.dil = DepthwiseDilatedConv(out_channels, out_channels, activation=nn.GELU)

    def forward(self, x, feature):
        down = self.down(x)
        c1 = self.conv1(torch.cat([down, feature], dim=1))
        c2 = self.conv2(c1)
        atts = self.dil(c2)
        return atts


class AttentionProc(nn.Module):
    def __init__(self, channels):
        super(AttentionProc, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, feature):
        conv1 = self.conv1(feature)
        conv2 = self.conv2(conv1) + conv1
        return conv2


class DepthwiseDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=3, num_dilation=4, step=2, activation=nn.GELU):
        super(DepthwiseDilatedConv, self).__init__()
        self.max_num = 1 + num_dilation * step
        self.step = step
        for d in range(1, self.max_num, self.step):  # step=2, 1,3,5,7
            self.add_module(
                'dilconv%s' % (d),
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same', dilation=d, groups=in_channels),
                    nn.BatchNorm2d(in_channels),
                    nn.GELU()
                )
            )
        # using depthwise dilated convolution and pointwise convolution
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels*4, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation()
        )
        self.dilation = num_dilation

    def forward(self, x):
        dils = []
        for d in range(1, self.max_num, self.step):
            dils.append(getattr(self, 'dilconv%s' % (d))(x))
        result = self.point_conv(torch.cat(dils, dim=1))
        return result


if __name__ == '__main__':
    x = torch.rand(1, 3, 480, 640)
    model = DCA(cout=1)
    result = model(x)
    summary(model)
