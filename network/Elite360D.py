import os
from depth_tool.network.B2F_Fusion import *
from depth_tool.modules.resnet import *
from depth_tool.modules.mobilenet import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from depth_tool.modules.efficientnet import EfficientNetB5
from depth_tool.modules.Swin_backbone import SwinB, SwinT
from depth_tool.modules.dilateformer import Dilateformer_T

import numpy as np

os.environ['TORCH_HOME'] = 'checkpoints'

Encoder = {'resnet18': resnet18,
           'resnet34': resnet34,
           'resnet50': resnet50,
           'effinetb5': EfficientNetB5,
           'swinb': SwinB,
           'swint': SwinT,
           'dilateformert': Dilateformer_T
           }


def to_tuple(size):
    if isinstance(size, (list, tuple)):
        return tuple(size)
    elif isinstance(size, (int, float)):
        return tuple((size, size))
    else:
        raise ValueError('Unsupport datatype: {}'.format(type(size)))


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class NoupSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(NoupSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        f = torch.cat([x, concat_with], dim=1)
        return self._net(f)


class ERP_encoder_Res(nn.Module):
    def __init__(self, backbone, channel):
        super(ERP_encoder_Res, self).__init__()

        pretrained_model = Encoder[backbone](pretrained=True)
        encoder = pretrained_model

        self.conv1 = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(True)
        self.layer1 = encoder.layer1  # 64
        self.layer2 = encoder.layer2  # 128
        self.layer3 = encoder.layer3  # 256
        self.layer4 = encoder.layer4  # 512
        if backbone == "resnet50":
            self.down = nn.Conv2d(512 * 4, channel, kernel_size=1, stride=1, padding=0)
        else:
            self.down = nn.Conv2d(512, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, erp_rgb):
        bs, c, erp_h, erp_w = erp_rgb.shape
        conv1 = self.relu(self.bn1(self.conv1(erp_rgb)))  # h/2 * w/2 * 64
        pool = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        x_downsample = []
        layer1 = self.layer1(pool)  # h/4 * w/4 * 64
        layer2 = self.layer2(layer1)  # h/8 * w/8 * 128
        layer3 = self.layer3(layer2)  # h/16 * w/16 * 256
        layer4 = self.layer4(layer3)  # h/32 * w/32 * 512
        layer4_reshape = self.down(layer4)  # h/32 * w/32 * embedding channel
        x_downsample.append(conv1)
        x_downsample.append(layer1)
        x_downsample.append(layer2)
        x_downsample.append(layer3)
        return layer4_reshape, x_downsample


class ERP_decoder_Res(nn.Module):
    def __init__(self, channel, features=512, backbone=0):
        super(ERP_decoder_Res, self).__init__()
        features = int(features)

        self.conv2 = nn.Conv2d(channel, features, kernel_size=1, stride=1, padding=0)
        if backbone == "resnet50":
            self.up1 = UpSampleBN(skip_input=features // 1 + 1024, output_features=features // 2)
            self.up2 = UpSampleBN(skip_input=features // 2 + 512, output_features=features // 4)
            self.up3 = UpSampleBN(skip_input=features // 4 + 256, output_features=features // 8)
            self.up4 = UpSampleBN(skip_input=features // 8 + 64, output_features=features // 16)
        else:
            self.up1 = UpSampleBN(skip_input=features // 1 + 256, output_features=features // 2)
            self.up2 = UpSampleBN(skip_input=features // 2 + 128, output_features=features // 4)
            self.up3 = UpSampleBN(skip_input=features // 4 + 64, output_features=features // 8)
            self.up4 = UpSampleBN(skip_input=features // 8 + 64, output_features=features // 16)

        self.out_conv = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, erp_feature, downsample_list):
        conv, layer1, layer2, layer3 = downsample_list[0], downsample_list[1], downsample_list[2], downsample_list[3]
        x_d0 = self.conv2(erp_feature.contiguous())
        x_d1 = self.up1(x_d0, layer3)
        x_d2 = self.up2(x_d1, layer2)
        x_d3 = self.up3(x_d2, layer1)
        x_d4 = self.up4(x_d3, conv)
        x = F.interpolate(x_d4, scale_factor=2, mode='nearest')
        out = self.out_conv(x)
        return out


class Elite360D_ResNet(nn.Module):
    def __init__(self, backbone, min_value=0, max_value=10, ico_nblocks=3,
                 ico_nneighbor=32, ico_level=4, embed_channel=32, resolution=512, L=1):
        super(Elite360D_ResNet, self).__init__()

        self.min_value = min_value
        self.max_value = max_value
        self.resolution = resolution
        self.embed_channel = embed_channel
        self.ico_level = ico_level
        self.ico_nblocks = ico_nblocks
        self.ico_nneighbor = ico_nneighbor
        self.backbone = backbone
        self.L = L
        ## Encoder
        self.ERP_encoder = ERP_encoder_Res(channel=self.embed_channel, backbone=self.backbone)
        self.EI_Fusion = EI_Adaptive_Fusion(resolution=self.resolution, ico_level=self.ico_level,
                                            ico_nblocks=self.ico_nblocks, ico_nneighbor=self.ico_nneighbor,
                                            embedding_dim=self.embed_channel, L=self.L)
        ## Decoder
        self.decoder = ERP_decoder_Res(channel=self.embed_channel, backbone=backbone)

        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, erp_rgb, ico_rgb, ico_coord):
        ## Encoder
        bs, _, erp_h, erp_w = erp_rgb.shape
        bottle_feature, downsample_list = self.ERP_encoder(erp_rgb)

        bottle_feature_, downsample_list_ = self.EI_Fusion(bottle_feature, downsample_list, ico_rgb,
                                                           ico_coord)  # bottle_feature, downsample_list#

        # Decoder
        y = self.decoder(bottle_feature_, downsample_list_)

        outputs = {}
        outputs["pred_depth"] = self.max_value * self.sigmoid(y)

        return outputs


class ERP_encoder_effb5(nn.Module):
    def __init__(self, backbone, channel):
        super(ERP_encoder_effb5, self).__init__()

        pretrained_model = Encoder[backbone](pretrained=True)
        self.encoder = pretrained_model

        self.down = nn.Conv2d(512, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, erp_rgb):
        bs, c, erp_h, erp_w = erp_rgb.shape
        endpoints = self.encoder(erp_rgb)
        x_downsample = []
        conv1 = endpoints["reduction_1"]  # h/2 * w/2 * 24
        layer1 = endpoints["reduction_2"]  # h/4 * w/4 * 40
        layer2 = endpoints["reduction_3"]  # h/8 * w/8 * 64
        layer3 = endpoints["reduction_4"]  # h/16 * w/16 * 176
        layer4 = endpoints["reduction_5"]  # h/32 * w/32 * 512
        layer4_reshape = self.down(layer4)  # h/32 * w/32 * embedding channel
        x_downsample.append(conv1)
        x_downsample.append(layer1)
        x_downsample.append(layer2)
        x_downsample.append(layer3)
        return layer4_reshape, x_downsample


class ERP_decoder_effb5(nn.Module):
    def __init__(self, channel, features=512):
        super(ERP_decoder_effb5, self).__init__()
        features = int(features)

        self.conv2 = nn.Conv2d(channel, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=features // 1 + 176, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 64, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 40, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 24, output_features=features // 16)

        self.out_conv = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, erp_feature, downsample_list):
        conv, layer1, layer2, layer3 = downsample_list[0], downsample_list[1], downsample_list[2], downsample_list[3]

        x_d0 = self.conv2(erp_feature.contiguous())

        x_d1 = self.up1(x_d0, layer3)
        x_d2 = self.up2(x_d1, layer2)
        x_d3 = self.up3(x_d2, layer1)
        x_d4 = self.up4(x_d3, conv)
        #         x_d5 = self.up5(x_d4, features[0])
        x = F.interpolate(x_d4, scale_factor=2, mode='nearest')
        out = self.out_conv(x)
        return out


class Elite360D_effb5(nn.Module):
    def __init__(self, backbone, min_value=0, max_value=10, ico_nblocks=3,
                 ico_nneighbor=32, ico_level=4, embed_channel=32, resolution=512, L=1):
        super(Elite360D_effb5, self).__init__()

        self.min_value = min_value
        self.max_value = max_value

        self.resolution = resolution
        self.embed_channel = embed_channel
        self.ico_level = ico_level
        self.ico_nblocks = ico_nblocks
        self.ico_nneighbor = ico_nneighbor
        self.backbone = backbone
        self.L = L
        ## Encoder
        self.ERP_encoder = ERP_encoder_effb5(channel=self.embed_channel, backbone=self.backbone)
        self.EI_Fusion = EI_Adaptive_Fusion(resolution=self.resolution, ico_level=self.ico_level,
                                            ico_nblocks=self.ico_nblocks, ico_nneighbor=self.ico_nneighbor,
                                            embedding_dim=self.embed_channel, L=self.L)
        ## Decoder
        self.decoder = ERP_decoder_effb5(channel=self.embed_channel)

        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, erp_rgb, ico_rgb, ico_coord):
        ## Encoder
        bs, _, erp_h, erp_w = erp_rgb.shape
        bottle_feature, downsample_list = self.ERP_encoder(erp_rgb)

        bottle_feature_, downsample_list_ = self.EI_Fusion(bottle_feature, downsample_list, ico_rgb, ico_coord)

        # Decoder
        y = self.decoder(bottle_feature_, downsample_list_)

        outputs = {}
        outputs["pred_depth"] = self.max_value * self.sigmoid(y)

        return outputs


class ERP_encoder_Res_Swin(nn.Module):
    def __init__(self, backbone, channel):
        super(ERP_encoder_Res_Swin, self).__init__()

        pretrained_model = Encoder[backbone](pretrained=True)
        self.encoder = pretrained_model

        self.down = nn.Conv2d(1024, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, erp_rgb):
        bs, c, erp_h, erp_w = erp_rgb.shape
        downsample_list = self.encoder(erp_rgb)
        x_downsample = []
        layer1 = downsample_list[0]  # h/4 * w/4 * 40
        layer2 = downsample_list[1]  # h/8 * w/8 * 64
        layer3 = downsample_list[2]  # h/16 * w/16 * 176
        layer4 = downsample_list[3]  # h/32 * w/32 * 512

        layer4_reshape = self.down(layer4.contiguous())  # h/32 * w/32 * embedding channel
        x_downsample.append(layer1)
        x_downsample.append(layer2)
        x_downsample.append(layer3)
        return layer4_reshape, x_downsample


class ERP_decoder_Res_Swin(nn.Module):
    def __init__(self, channel, features=512):
        super(ERP_decoder_Res_Swin, self).__init__()
        features = int(features)

        self.conv2 = nn.Conv2d(channel, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=features // 1 + 512, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 256, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 128, output_features=features // 8)
        self.out_conv = nn.Conv2d(features // 8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, erp_feature, downsample_list):
        layer1, layer2, layer3 = downsample_list[0], downsample_list[1], downsample_list[2]

        x_d0 = self.conv2(erp_feature.contiguous())

        x_d1 = self.up1(x_d0, layer3)
        x_d2 = self.up2(x_d1, layer2)
        x_d3 = self.up3(x_d2, layer1)
        #         x_d5 = self.up5(x_d4, features[0])
        x = F.interpolate(x_d3, scale_factor=4, mode='nearest')
        out = self.out_conv(x)
        return out


class Elite360D_SwinB(nn.Module):
    def __init__(self, backbone, min_value=0, max_value=10, ico_nblocks=3,
                 ico_nneighbor=32, ico_level=4, embed_channel=32, resolution=512, L=1):
        super(Elite360D_SwinB, self).__init__()

        self.min_value = min_value
        self.max_value = max_value

        self.resolution = resolution
        self.embed_channel = embed_channel
        self.ico_level = ico_level
        self.ico_nblocks = ico_nblocks
        self.ico_nneighbor = ico_nneighbor
        self.backbone = backbone
        self.L = L
        self.ERP_encoder = ERP_encoder_Res_Swin(channel=self.embed_channel, backbone=self.backbone)
        self.EI_Fusion = EI_Adaptive_Fusion(resolution=self.resolution, ico_level=self.ico_level,
                                            ico_nblocks=self.ico_nblocks,
                                            ico_nneighbor=self.ico_nneighbor, embedding_dim=self.embed_channel,
                                            L=self.L)
        ## Decoder
        self.decoder = ERP_decoder_Res_Swin(channel=self.embed_channel)

        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, erp_rgb, ico_rgb, ico_coord):
        ## Encoder
        bs, _, erp_h, erp_w = erp_rgb.shape
        bottle_feature, downsample_list = self.ERP_encoder(erp_rgb)

        bottle_feature_, downsample_list_ = self.EI_Fusion(bottle_feature, downsample_list, ico_rgb, ico_coord)

        # Decoder
        y = self.decoder(bottle_feature_, downsample_list_)

        outputs = {}
        outputs["pred_depth"] = self.max_value * self.sigmoid(y)

        return outputs


class ERP_encoder_Res_SwinT(nn.Module):
    def __init__(self, backbone, channel):
        super(ERP_encoder_Res_SwinT, self).__init__()

        pretrained_model = Encoder[backbone](pretrained=True)
        self.encoder = pretrained_model

        self.down = nn.Conv2d(768, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, erp_rgb):
        bs, c, erp_h, erp_w = erp_rgb.shape
        m = erp_h // 256
        n = erp_w // 256
        input = rearrange(erp_rgb, 'b c (m hs) (n ws)-> (b m n) c hs ws', hs=256, ws=256)
        downsample_list = self.encoder(input)
        x_downsample = []
        layer1 = rearrange(downsample_list[0], '(b m n) (hs ws) c -> b c  (m hs) (n ws)', hs=256 // 4, ws=256 // 4, m=m,
                           n=n)  # h/4 * w/4 * 96
        layer2 = rearrange(downsample_list[1], '(b m n) (hs ws) c -> b c  (m hs) (n ws)', hs=256 // 8, ws=256 // 8, m=m,
                           n=n)  # h/8 * w/8 * 192
        layer3 = rearrange(downsample_list[2], '(b m n) (hs ws) c -> b c  (m hs) (n ws)', hs=256 // 16, ws=256 // 16,
                           m=m, n=n)  # h/16 * w/16 * 384
        layer4 = rearrange(downsample_list[3], '(b m n) (hs ws) c -> b c  (m hs) (n ws)', hs=256 // 32, ws=256 // 32,
                           m=m, n=n)  # h/32 * w/32 * 768
        layer5 = rearrange(downsample_list[4], '(b m n) (hs ws) c -> b c  (m hs) (n ws)', hs=256 // 32, ws=256 // 32,
                           m=m, n=n)  # h/32 * w/32 * 768
        layer5_reshape = self.down(layer5.contiguous())  # h/32 * w/32 * embedding channel
        x_downsample.append(layer1)
        x_downsample.append(layer2)
        x_downsample.append(layer3)
        x_downsample.append(layer4)
        return layer5_reshape, x_downsample


class ERP_decoder_Res_SwinT(nn.Module):
    def __init__(self, channel, features=512):
        super(ERP_decoder_Res_SwinT, self).__init__()
        features = int(features)

        self.conv2 = nn.Conv2d(channel, features, kernel_size=1, stride=1, padding=0)
        self.up1 = NoupSampleBN(skip_input=features // 1 + 768, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 384, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 192, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 96, output_features=features // 16)
        self.out_conv = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, erp_feature, downsample_list):
        layer1, layer2, layer3, layer4 = downsample_list[0], downsample_list[1], downsample_list[2], downsample_list[3]

        x_d0 = self.conv2(erp_feature.contiguous())

        x_d1 = self.up1(x_d0, layer4)
        x_d2 = self.up2(x_d1, layer3)
        x_d3 = self.up3(x_d2, layer2)
        x_d4 = self.up4(x_d3, layer1)
        #         x_d5 = self.up5(x_d4, features[0])
        x = F.interpolate(x_d4, scale_factor=4, mode='nearest')
        out = self.out_conv(x)
        return out


class Elite360D_SwinT(nn.Module):
    def __init__(self, backbone, min_value=0, max_value=10, ico_nblocks=3,
                 ico_nneighbor=32, ico_level=4, embed_channel=32, resolution=512, L=1):
        super(Elite360D_SwinT, self).__init__()

        self.min_value = min_value
        self.max_value = max_value

        self.resolution = resolution
        self.embed_channel = embed_channel
        self.ico_level = ico_level
        self.ico_nblocks = ico_nblocks
        self.ico_nneighbor = ico_nneighbor
        self.backbone = backbone
        self.L = L
        ## Encoder
        self.ERP_encoder = ERP_encoder_Res_SwinT(channel=self.embed_channel, backbone=self.backbone)
        self.EI_Fusion = EI_Adaptive_Fusion(resolution=self.resolution, ico_level=self.ico_level,
                                            ico_nblocks=self.ico_nblocks,
                                            ico_nneighbor=self.ico_nneighbor, embedding_dim=self.embed_channel,
                                            L=self.L)
        ## Decoder
        self.decoder = ERP_decoder_Res_SwinT(channel=self.embed_channel)

        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, erp_rgb, ico_rgb, ico_coord):
        ## Encoder
        bs, _, erp_h, erp_w = erp_rgb.shape
        bottle_feature, downsample_list = self.ERP_encoder(erp_rgb)

        bottle_feature_, downsample_list_ = self.EI_Fusion(bottle_feature, downsample_list, ico_rgb, ico_coord)

        # Decoder
        y = self.decoder(bottle_feature_, downsample_list_)

        outputs = {}
        outputs["pred_depth"] = self.max_value * self.sigmoid(y)

        return outputs


class ERP_encoder_Res_DilateT(nn.Module):
    def __init__(self, backbone, channel):
        super(ERP_encoder_Res_DilateT, self).__init__()

        pretrained_model = Encoder[backbone](pretrained=True)
        self.encoder = pretrained_model

        self.down = nn.Conv2d(576, channel, kernel_size=1, stride=1, padding=0)

    def forward(self, erp_rgb):
        bs, c, erp_h, erp_w = erp_rgb.shape
        layer4, downsample_list = self.encoder(erp_rgb)
        x_downsample = []
        layer1 = downsample_list[0]  # h/4 * w/4 * 72
        layer2 = downsample_list[1]  # h/8 * w/8 * 144
        layer3 = downsample_list[2]  # h/16 * w/16 * 288

        layer4_reshape = self.down(layer4)  # h/32 * w/32 * embedding channel
        x_downsample.append(layer1)
        x_downsample.append(layer2)
        x_downsample.append(layer3)
        return layer4_reshape, x_downsample


class ERP_decoder_Res_DilateT(nn.Module):
    def __init__(self, channel, features=512):
        super(ERP_decoder_Res_DilateT, self).__init__()
        features = int(features)

        self.conv2 = nn.Conv2d(channel, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=features // 1 + 288, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 144, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 72, output_features=features // 8)
        self.out_conv = nn.Conv2d(features // 8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, erp_feature, downsample_list):
        layer1, layer2, layer3 = downsample_list[0], downsample_list[1], downsample_list[2]

        x_d0 = self.conv2(erp_feature.contiguous())

        x_d1 = self.up1(x_d0, layer3)
        x_d2 = self.up2(x_d1, layer2)
        x_d3 = self.up3(x_d2, layer1)
        #         x_d5 = self.up5(x_d4, features[0])
        x = F.interpolate(x_d3, scale_factor=4, mode='nearest')
        out = self.out_conv(x)
        return out


class Elite360D_DilateT(nn.Module):
    def __init__(self, backbone, min_value=0, max_value=10, ico_nblocks=3,
                 ico_nneighbor=32, ico_level=4, embed_channel=32, resolution=512, L=1):
        super(Elite360D_DilateT, self).__init__()

        self.min_value = min_value
        self.max_value = max_value
        self.resolution = resolution
        self.embed_channel = embed_channel
        self.ico_level = ico_level
        self.ico_nblocks = ico_nblocks
        self.ico_nneighbor = ico_nneighbor
        self.backbone = backbone
        self.L = L
        ## Encoder
        self.ERP_encoder = ERP_encoder_Res_DilateT(channel=self.embed_channel, backbone=self.backbone)
        self.EI_Fusion = EI_Adaptive_Fusion(resolution=self.resolution, ico_level=self.ico_level,
                                            ico_nblocks=self.ico_nblocks,
                                            ico_nneighbor=self.ico_nneighbor, embedding_dim=self.embed_channel,
                                            L=self.L)
        ## Decoder
        self.decoder = ERP_decoder_Res_DilateT(channel=self.embed_channel)

        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, erp_rgb, ico_rgb, ico_coord):
        ## Encoder
        bs, _, erp_h, erp_w = erp_rgb.shape
        bottle_feature, downsample_list = self.ERP_encoder(erp_rgb)

        bottle_feature_, downsample_list_ = self.EI_Fusion(bottle_feature, downsample_list, ico_rgb, ico_coord)

        # Decoder
        y = self.decoder(bottle_feature_, downsample_list_)

        outputs = {}
        outputs["pred_depth"] = self.max_value * self.sigmoid(y)

        return outputs
