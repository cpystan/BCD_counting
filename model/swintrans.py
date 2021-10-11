# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.swin_transformer import swin_base_patch4_window12_384_in22k, _create_swin_transformer, SwinTransformer, \
    build_model_with_cfg, checkpoint_filter_fn, overlay_external_default_cfg, default_cfgs
from copy import deepcopy
from timm.models.helpers import load_pretrained, load_custom_pretrained


class VisionTransformer_token(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        x = self.output1(x)

        return x


class VisionTransformer_gap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, 1:]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        x = F.adaptive_avg_pool1d(x, (48))
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.output1(x)
        return x

# def _create_swin_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
#     if default_cfg is None:
#         default_cfg = deepcopy(default_cfgs[variant])
#     overlay_external_default_cfg(default_cfg, kwargs)
#     default_num_classes = default_cfg['num_classes']
#     default_img_size = default_cfg['input_size'][-2:]
#
#     num_classes = kwargs.pop('num_classes', default_num_classes)
#     img_size = kwargs.pop('img_size', default_img_size)
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')
#     feature_cfg = {'feature_cls': 'hook'}
#
#     model = build_model_with_cfg(
#         SwinTransformer, variant, pretrained,
#         default_cfg=default_cfg,
#         feature_cfg = feature_cfg,
#         img_size=img_size,
#         num_classes=num_classes,
#         pretrained_filter_fn=checkpoint_filter_fn,
#         **kwargs)
#
#     return model

# class SwinTransformer_gap(SwinTransformer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         num_patches = self.patch_embed.num_patches
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
#         'swin_base_patch4_window12_384_in22k'
#         super().__init__(*args, **kwargs)
#         self.num_classes = 1
#
#         self.model_kwargs = dict(
#             patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
#         # swin_large_patch4_window12_384
#         # swin_base_patch4_window12_384
#         # swin_base_patch4_window12_384_in22k
#         self.forward_features = _create_swin_transformer('swin_base_patch4_window12_384_in22k',
#                                                          pretrained=True, **self.model_kwargs)
#         print('type', type(self.forward_features))
#         self.head = nn.Linear(self.num_features, self.num_classes)
#
#         trunc_normal_(self.pos_embed, std=.02)
#
#         # self.output1 = nn.Sequential(
#         #     nn.ReLU(),
#         #     nn.Linear(21841, 128),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.5),
#         #     nn.Linear(128, 1)
#         # )
#         # self.output1.apply(self._init_weights)
#
#     def forward(self, x):
#         feature_map = self.forward_features(x)
#         out = self.head(feature_map)
#         # print('forward_features out', x.shape)
#         # x = F.adaptive_avg_pool1d(x, (48))
#         # print('adaptive_avg_pool1d out', x.shape)
#         # x = x.view(x.shape[0], -1)
#         # x = self.output1(x)
#         return out, feature_map

class SwinTransformer_gap(SwinTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 1
        self.patch_size = 4
        self.window_size = 12
        self.embed_dim = 128
        self.depths = (2, 2, 18, 2)
        self.num_heads = (4, 8, 16, 32)
        self.head = nn.Linear(self.num_features, self.num_classes)


    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        feature_att = self.forward_features(x)
        x = self.head(feature_att)
        return x

@register_model
def swin_base_gap(pretrained=False, variant = 'swin_base_patch4_window12_384_in22k', pretrained_custom_path = None):

    model_kwargs = dict(patch_size=4, window_size=12, embed_dim=128,
                        depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))

    default_cfg = deepcopy(default_cfgs[variant])
    default_cfg.setdefault('architecture', variant)
    default_num_classes = default_cfg['num_classes']
    num_classes = model_kwargs.pop('num_classes', default_num_classes)

    model = SwinTransformer_gap()
    model.default_cfg = default_cfg
    if pretrained:
        if not pretrained_custom_path:
            load_custom_pretrained(pretrained_custom_path)
        else:
            load_pretrained(model, num_classes=num_classes,
                            in_chans=model_kwargs.get('in_chans', 3),
                            filter_fn= None, strict= False)

    return model


@register_model
def base_patch16_384_token(pretrained=False, **kwargs):
    model = VisionTransformer_token(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        # checkpoint = torch.load(
        #     './Networks/deit_base_patch16_384-8de9b5d1.pth')
        checkpoint = torch.load(
            "/data2/Public_dataset/deit_base_patch16_384-8de9b5d1.pth")
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model


@register_model
def base_patch16_384_gap(pretrained=False, **kwargs):
    model = VisionTransformer_gap(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        # checkpoint = torch.load(
        #     './Networks/deit_base_patch16_384-8de9b5d1.pth')
        checkpoint = torch.load(
            "/data2/Public_dataset/deit_base_patch16_384-8de9b5d1.pth")
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model

