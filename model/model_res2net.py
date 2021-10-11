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
from timm.models.layers.classifier import create_classifier, _create_fc
import math
import pdb

class Attention_2class(Bottle2neck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_pos_mask(self, pos_attn, attn, margin):
        # print('------pos_attn_info--------')
        # print('pos_attn_shape', pos_attn.size(),
        #       'pos_attn_max', torch.max(pos_attn),
        #       'pos_attn_min', torch.min(pos_attn),
        #       'pos_attn_mean', torch.mean(pos_attn))
        #
        # print('------attn__info--------')
        # print('neg_attn_shape', attn.size(),
        #       'neg_attn_max', torch.max(attn),
        #       'neg_attn_min', torch.min(attn),
        #       'neg_attn_mean', torch.mean(attn))
        # pdb.set_trace()
        out = torch.maximum(torch.tensor(0).cuda(), attn - pos_attn + torch.tensor(margin).cuda())
        return out

    def forward(self, x, pos_attn = None, margin = 3):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        attn = out

        if pos_attn is not None:
            out = self.apply_pos_mask(pos_attn, attn, margin)

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

        out = out+ shortcut
        out = self.relu(out)

        return out, attn

    # def forward(self, x):
    #     pos_feature, pos_attn = self.forward_features(x)
    #     neg_feature, _ = self.forward_features(x, pos_attn = pos_attn)
    #
    #     return pos_feature, neg_feature


class Res2Net_2class(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 1
        self.pool_type = 'avg'
        self.drop_rate = 0.0
        self.use_conv = False
        self.zero_init_last_bn = True
        self.bottlenet_num = 3
        self.downsample = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.pos_atten_2class1 = Attention_2class(inplanes=1024, planes=512, stride = 2, norm_layer=nn.BatchNorm2d,
                                                  downsample = self.downsample)
        self.pos_atten_2class2 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)
        self.pos_atten_2class3 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)

        self.neg_atten_2class1 = Attention_2class(inplanes=1024, planes=512, stride=2,norm_layer=nn.BatchNorm2d,
                                                  downsample=self.downsample)
        self.neg_atten_2class2 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)
        self.neg_atten_2class3 = Attention_2class(inplanes=2048, planes=512, norm_layer=nn.BatchNorm2d)

        self.global_pool_pos, self.fc_pos = create_classifier(self.num_features, self.num_classes,
                                                              pool_type=self.pool_type)
        self.global_pool_neg, self.fc_neg = create_classifier(self.num_features, self.num_classes,
                                                              pool_type=self.pool_type)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4[:-1](x)
        # print('00----------------------------')
        # print(self.layer4)
        # print(self.atten_2class1)
        # print(self.atten_2class2)
        # print(self.atten_2class3)
        # pdb.set_trace()

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # pos_feature, neg_feature = self.atten_2class(x)
        pos_feature, attn1 = self.pos_atten_2class1(x)
        pos_feature, attn2 = self.pos_atten_2class2(pos_feature)
        pos_feature, attn3 = self.pos_atten_2class3(pos_feature)
        pos_feature = self.global_pool_pos(pos_feature)
        pos_feature = F.dropout(pos_feature, p=float(self.drop_rate), training=self.training)
        pos_count = self.fc_pos(pos_feature)

        neg_feature, _ = self.neg_atten_2class1(x, pos_attn = attn1)
        neg_feature, _ = self.neg_atten_2class2(neg_feature, pos_attn = attn2)
        neg_feature, _ = self.neg_atten_2class3(neg_feature, pos_attn = attn3)
        neg_feature = self.global_pool_neg(neg_feature)
        neg_feature = F.dropout(neg_feature, p=float(self.drop_rate), training=self.training)
        neg_count = self.fc_neg(neg_feature)
        return pos_count, neg_count

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