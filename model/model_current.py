import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, SelectAdaptivePool2d
from timm.models.res2net import Bottle2neck, default_cfgs, Bottle2neck, ResNet
from copy import deepcopy
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.layers.classifier import create_classifier, _create_fc, SelectAdaptivePool2d
import math
import pdb
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy


class Attention_2class(Bottle2neck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_pos_mask(self, pos_attn, attn, ratio):
        margin = 0
        # print('------pos_attn_info--------')
        # print('pos_attn_shape', pos_attn.size(),'\n',
        #       'pos_attn_max', torch.max(pos_attn),'\n',
        #       'pos_attn_min', torch.min(pos_attn),'\n',
        #       'pos_attn_mean', torch.mean(pos_attn))
        #
        # pos_feature_map = torch.mean(pos_attn, dim = 1)[0]
        # pos_feature_map = pos_feature_map.cpu().detach().numpy()
        # pos_feature_map = pos_feature_map / np.max(pos_feature_map) * 255
        # plt.figure(2)
        # plt.imshow(pos_feature_map, cmap='gray')
        # plt.savefig('DR_pos.png')
        # plt.show()
        #
        # print('------attn__info--------')
        # print('neg_attn_shape', attn.size(),'\n',
        #       'neg_attn_max', torch.max(attn),'\n',
        #       'neg_attn_min', torch.min(attn),'\n',
        #       'neg_attn_mean', torch.mean(attn))
        # neg_feature_map = torch.mean(attn, dim=1)[0]
        # neg_feature_map = neg_feature_map.cpu().detach().numpy()
        # neg_feature_map = neg_feature_map / np.max(neg_feature_map) * 255
        # plt.figure(3)
        # plt.imshow(neg_feature_map, cmap='gray')
        # plt.savefig('DR_neg.png')
        # plt.show()

        out = torch.maximum(
            torch.tensor(0).cuda(),
            attn - ratio * torch.mean(pos_attn, dim= 1).unsqueeze(1) + torch.tensor(margin).cuda())
        # print('------out_info--------')
        # print('out_shape', out.size(),'\n',
        #       'out_max', torch.max(out),'\n',
        #       'out_min', torch.min(out),'\n',
        #       'out_mean', torch.mean(out))
        # out_feature_map = torch.mean(out, dim=1)[0]
        # out_feature_map = out_feature_map.cpu().detach().numpy()
        # out_feature_map = out_feature_map / np.max(out_feature_map) * 255
        # plt.figure(4)
        # plt.imshow(out_feature_map, cmap='gray')
        # plt.savefig('DR_out.png')
        # plt.show()
        # pdb.set_trace()

        return out

    def forward(self, x, pos_attn = None, ratio = 0):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        spo = []
        sp = spx[0]  # redundant, for torchscript
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.is_first:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(sp)
            sp = self.relu(sp)
            spo.append(sp)
        if self.scale > 1:
            if self.pool is not None:
                # self.is_first == True, None check for torchscript
                spo.append(self.pool(spx[-1]))
            else:
                spo.append(spx[-1])
        out = torch.cat(spo, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        attn = out

        if pos_attn is not None:
            out = self.apply_pos_mask(pos_attn, out, ratio=ratio)


        return out, attn


class Res2Net_2class(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 1
        self.pool_type = 'avg'
        self.drop_rate = 0.0
        self.use_conv = False
        self.zero_init_last_bn = True

        # self.downsample_layer1 = nn.Sequential(nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #                                        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        #
        # self.pos_atten_2class1 = Attention_2class(inplanes = 64, planes = 64, norm_layer=nn.BatchNorm2d,
        #                                           downsample = self.downsample_layer1)
        # self.pos_atten_2class2 = Attention_2class(inplanes = 256, planes = 64, norm_layer=nn.BatchNorm2d)
        # self.pos_atten_2class3 = Attention_2class(inplanes = 256, planes = 64, norm_layer=nn.BatchNorm2d)
        #
        #
        # self.neg_atten_2class1 = Attention_2class(inplanes = 64, planes = 64, norm_layer=nn.BatchNorm2d,
        #                                           downsample = self.downsample_layer1)
        # self.neg_atten_2class2 = Attention_2class(inplanes = 256, planes = 64, norm_layer=nn.BatchNorm2d)
        # self.neg_atten_2class3 = Attention_2class(inplanes = 256, planes = 64, norm_layer=nn.BatchNorm2d)
        #
        self.downsample = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        # self.downsample = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
        #                                 nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True,
        #                                                track_running_stats=True))

        self.pos_layer4 = nn.Sequential(Bottle2neck(inplanes=1024, planes=512, stride = 2, norm_layer=nn.BatchNorm2d,
                                                  downsample = self.downsample),
                                        Bottle2neck(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d),
                                        Bottle2neck(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d))

        self.neg_layer4 = nn.Sequential(Bottle2neck(inplanes=1024, planes=512, stride = 2, norm_layer=nn.BatchNorm2d,
                                                  downsample = self.downsample),
                                        Bottle2neck(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d),
                                        Bottle2neck(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d))

        self.global_pool_pos, self.fc_pos = create_classifier(self.num_features, self.num_classes,
                                                              pool_type=self.pool_type)
        self.global_pool_neg, self.fc_neg = create_classifier(self.num_features, self.num_classes,
                                                              pool_type=self.pool_type)

        self.layer5 = copy.deepcopy(self.layer1)
        self.layer6 = copy.deepcopy(self.layer2)
        self.layer7 = copy.deepcopy(self.layer3)
        self.layer8 = copy.deepcopy(self.layer4)

        # self.pos_layer4_atten1 = Attention_2class(inplanes=1024, planes=512, stride = 2, norm_layer=nn.BatchNorm2d,
        #                                           downsample = self.downsample)
        # self.pos_layer4_atten2 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)
        # self.pos_layer4_atten3 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)
        #
        # self.neg_layer4_atten1 = Attention_2class(inplanes=1024, planes=512, stride = 2, norm_layer=nn.BatchNorm2d,
        #                                           downsample = self.downsample)
        # self.neg_layer4_atten2 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)
        # self.neg_layer4_atten3 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)

        # self.pos_layer4_atten1 = Attention_2class(inplanes=2048, planes=512, stride=2, norm_layer=nn.BatchNorm2d,
        #                                           downsample=self.downsample)
        # self.pos_layer4_atten2 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)
        # self.pos_layer4_atten3 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)
        #
        # self.neg_layer4_atten1 = Attention_2class(inplanes=2048, planes=512, stride=2, norm_layer=nn.BatchNorm2d,
        #                                           downsample=self.downsample)
        # self.neg_layer4_atten2 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)
        # self.neg_layer4_atten3 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)






        # self.layer7 = copy.deepcopy(self.layer4)

        # self.layer8 = copy.deepcopy(self.layer2)
        # self.layer9 = copy.deepcopy(self.layer3)
        # self.layer10 = copy.deepcopy(self.layer4)

        # self.global_pool = SelectAdaptivePool2d(pool_type = self.pool_type, flatten = nn.Flatten(start_dim=1, end_dim=-1))



    def forward_features_(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return x

    def share_features(self, pos_feature, neg_feature):
        # x = self.layer1(x)
        # pos_feature = self.layer1(pos_feature)
        # pos_feature = self.layer2(pos_feature)
        pos_feature = self.layer3(pos_feature)
        # pos_feature = self.layer4(pos_feature)

        # neg_feature = self.layer5(neg_feature)
        # neg_feature = self.layer6(neg_feature)
        neg_feature = self.layer7(neg_feature)
        # neg_feature = self.layer8(neg_feature)

        return pos_feature, neg_feature

    # def atten_2class(self, x, mask = False, pos_attn1 = None, pos_attn2 = None, pos_attn3 = None, ratio = 0):
    #     if mask == False:
    #         x, attn1 = self.pos_atten_2class1(x)
    #         x, attn2 = self.pos_atten_2class2(x)
    #         x, attn3 = self.pos_atten_2class3(x)
    #         return x, attn1, attn2, attn3
    #     else:
    #         x, _ = self.neg_atten_2class1(x, pos_attn1, ratio = ratio)
    #         x, _ = self.neg_atten_2class2(x, pos_attn2, ratio = ratio * 1/3)
    #         x, _ = self.neg_atten_2class3(x, pos_attn3, ratio = ratio * 2/3)
    #         return x

    # def atten_2class(self, x, mask = False, pos_attn1 = None, pos_attn2 = None, pos_attn3 = None, ratio = 0):
    #     if mask == False:
    #         x, attn1 = self.pos_atten_2class1(x)
    #         x, attn2 = self.pos_atten_2class2(x)
    #         x, attn3 = self.pos_atten_2class3(x)
    #         return x, attn1, attn2, attn3
    #     else:
    #
    #         x, attn1 = self.neg_atten_2class1(x)
    #         x, attn2 = self.neg_atten_2class2(x)
    #         x, attn3 = self.neg_atten_2class3(x)
    #         return x, attn1, attn2, attn3

    # def atten_layer4(self, x, mask = False, pos_attn1 = None, pos_attn2 = None, pos_attn3 = None, ratio = 0):
    #     if mask == False:
    #         x, attn1 = self.pos_layer4_atten1(x)
    #         x, attn2 = self.pos_layer4_atten2(x)
    #         x, attn3 = self.pos_layer4_atten3(x)
    #         return x, attn1, attn2, attn3
    #     else:
    #         x, _ = self.neg_layer4_atten1(x, pos_attn1, ratio = ratio)
    #         x, _ = self.neg_layer4_atten2(x, pos_attn2, ratio = ratio * 1/3)
    #         x, _ = self.neg_layer4_atten3(x, pos_attn3, ratio = ratio * 2/3)
    #         return x

    def forward(self, x, ratio):

        # x = self.forward_features_(x)
        # pos_feature, attn1, attn2, attn3 = self.atten_2class(x, mask = False)
        # neg_feature = self.atten_2class(x, mask= True, pos_attn1 = attn1, pos_attn2 = attn2, pos_attn3 = attn3, ratio = ratio)
        #
        # pos_feature, neg_feature = self.share_features(pos_feature, neg_feature)
        #
        # pos_feature = self.pos_layer4(pos_feature)
        # pos_feature = self.global_pool_pos(pos_feature)
        # pos_feature = F.dropout(pos_feature, p=float(self.drop_rate), training=self.training)
        # pos_count = self.fc_pos(pos_feature)
        #
        # neg_feature = self.neg_layer4(neg_feature)
        # neg_feature = self.global_pool_neg(neg_feature)
        # neg_feature = F.dropout(neg_feature, p=float(self.drop_rate), training=self.training)
        # neg_count = self.fc_neg(neg_feature)

        ## -------------------------------------------------------------

        x = self.forward_features_(x)
        pos_feature = self.layer2(x)
        neg_feature = self.layer6(x)

        attn_pos = pos_feature.detach()
        attn_neg = neg_feature

        pos_feature, neg_feature = self.share_features(pos_feature, neg_feature)

        pos_feature = self.pos_layer4(pos_feature)
        pos_feature = self.global_pool_pos(pos_feature)
        pos_feature = F.dropout(pos_feature, p=float(self.drop_rate), training=self.training)
        pos_count = self.fc_pos(pos_feature)

        neg_feature = self.neg_layer4(neg_feature)
        neg_feature = self.global_pool_neg(neg_feature)
        neg_feature = F.dropout(neg_feature, p=float(self.drop_rate), training=self.training)
        neg_count = self.fc_neg(neg_feature)
        # #

        ## -------------------------------------------------------------

        # x = self.forward_features_(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)

        # pos_feature, attn1, attn2, attn3 = self.atten_2class(x, mask=False)
        # neg_feature = self.atten_2class(x, mask=True, pos_attn1=attn1, pos_attn2=attn2, pos_attn3=attn3, ratio=ratio)

        # pos_feature, neg_feature = self.share_features(pos_feature, neg_feature)

        # pos_feature = self.pos_layer4(x)
        # pos_feature = self.global_pool_pos(pos_feature)
        # pos_feature = F.dropout(pos_feature, p=float(self.drop_rate), training=self.training)
        # pos_count = self.fc_pos(pos_feature)
        #
        # neg_feature = self.neg_layer4(x)
        # neg_feature = self.global_pool_neg(neg_feature)
        # neg_feature = F.dropout(neg_feature, p=float(self.drop_rate), training=self.training)
        # neg_count = self.fc_neg(neg_feature)

        ## -------------------------------------------------------------

        # x = self.forward_features_(x)
        #
        # pos_feature = self.layer1(x)
        # neg_feature = self.layer5(x)
        # # neg_feature = torch.maximum(torch.tensor(0).cuda(),
        # #                             self.layer5(x) - ratio * torch.mean(pos_feature, dim=1).unsqueeze(1))
        # # neg_feature = (self.layer5(x) - pos_feature)
        #
        # # pos_feature, attn1, attn2, attn3 = self.atten_2class(x, mask=False)
        # # neg_feature = self.atten_2class(x, mask=True, pos_attn1=attn1, pos_attn2=attn2, pos_attn3=attn3, ratio=ratio)
        #
        # pos_feature, neg_feature = self.share_features(pos_feature, neg_feature)
        #
        # pos_feature = self.pos_layer4(pos_feature)
        # pos_feature = self.global_pool_pos(pos_feature)
        # pos_feature = F.dropout(pos_feature, p=float(self.drop_rate), training=self.training)
        # pos_count = self.fc_pos(pos_feature)
        #
        # neg_feature = self.neg_layer4(neg_feature)
        # neg_feature = self.global_pool_neg(neg_feature)
        # neg_feature = F.dropout(neg_feature, p=float(self.drop_rate), training=self.training)
        # neg_count = self.fc_neg(neg_feature)

        ## -------------------------------------------------------------

        # x = self.forward_features_(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # # neg_feature = torch.maximum(torch.tensor(0).cuda(),
        # #                             self.layer5(x) - ratio * torch.mean(pos_feature, dim=1).unsqueeze(1))
        # # neg_feature = (self.layer5(x) - pos_feature)
        #
        # # pos_feature, attn1, attn2, attn3 = self.atten_2class(x, mask=False)
        # # neg_feature = self.atten_2class(x, mask=True, pos_attn1=attn1, pos_attn2=attn2, pos_attn3=attn3, ratio=ratio)
        #
        # # pos_feature, neg_feature = self.share_features(pos_feature, neg_feature)
        #
        # pos_feature = self.pos_layer4(x)
        # pos_feature = self.global_pool_pos(pos_feature)
        # pos_feature = F.dropout(pos_feature, p=float(self.drop_rate), training=self.training)
        # pos_count = self.fc_pos(pos_feature)
        #
        # neg_feature = self.neg_layer4(x)
        # neg_feature = self.global_pool_neg(neg_feature)
        # neg_feature = F.dropout(neg_feature, p=float(self.drop_rate), training=self.training)
        # neg_count = self.fc_neg(neg_feature)

        ## -------------------------------------------------------------

        # x = self.forward_features_(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # pos_feature, attn1, attn2, attn3 = self.atten_layer4(x, mask=False)
        # neg_feature = self.atten_layer4(x, mask=True, pos_attn1=attn1, pos_attn2=attn2, pos_attn3=attn3, ratio=ratio)
        # #
        # # neg_feature = self.atten_2class(x, mask= True, pos_attn1 = attn1, pos_attn2 = attn2, pos_attn3 = attn3, ratio = ratio)
        # #
        # # neg_feature = torch.maximum(torch.tensor(0).cuda(),
        # #                             self.layer5(x) - ratio * torch.mean(pos_feature, dim=1).unsqueeze(1))
        # # neg_feature = (self.layer5(x) - pos_feature)
        #
        # # pos_feature, attn1, attn2, attn3 = self.atten_2class(x, mask=False)
        # # neg_feature = self.atten_2class(x, mask=True, pos_attn1=attn1, pos_attn2=attn2, pos_attn3=attn3, ratio=ratio)
        #
        # # pos_feature, neg_feature = self.share_features(pos_feature, neg_feature)
        #
        # # pos_feature = self.pos_layer4(x)
        # pos_feature = self.global_pool_pos(pos_feature)
        # pos_feature = F.dropout(pos_feature, p=float(self.drop_rate), training=self.training)
        # pos_count = self.fc_pos(pos_feature)
        #
        # # neg_feature = self.neg_layer4(x)
        # neg_feature = self.global_pool_neg(neg_feature)
        # neg_feature = F.dropout(neg_feature, p=float(self.drop_rate), training=self.training)
        # neg_count = self.fc_neg(neg_feature)

        return pos_count, neg_count, attn_pos, attn_neg

def res2net_2class(pretrained = False, variant = 'res2net101_26w_4s',pretrained_custom_path = None):
    model_kwargs = dict(block=Bottle2neck, layers=[3, 4, 23, 3], base_width=26, block_args=dict(scale=4))


    default_cfg = deepcopy(default_cfgs[variant])
    default_cfg.setdefault('architecture', variant)
    default_num_classes = default_cfg['num_classes']
    num_classes = model_kwargs.pop('num_classes', default_num_classes)

    model = Res2Net_2class(block=Bottle2neck, layers=[3, 4, 23, 3], base_width=26, block_args=dict(scale=4))
    model.default_cfg = default_cfg
    if pretrained:
        if not pretrained_custom_path:
            load_custom_pretrained(pretrained_custom_path)
        else:
            load_pretrained(model, num_classes=num_classes,
                            in_chans=model_kwargs.get('in_chans', 3),
                            filter_fn=None, strict=False)

    return model