# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transformer import SnippetEmbedding
import yaml

with open("./config/anet.yaml", 'r', encoding='utf-8') as f:
        tmp = f.read()
        config = yaml.load(tmp, Loader=yaml.FullLoader)



class GSM(nn.Module):
    def __init__(self):
        super(GSM, self).__init__()
        self.len_feat = config['model']['feat_dim']
        self.temporal_scale = config['model']['temporal_scale']
        self.temp_scale = config['training']['scale']
        self.num_classes = config['dataset']['num_classes']+1
        self.n_heads = config['model']['embedding_head']
        self.embedding = SnippetEmbedding(self.n_heads, self.len_feat, self.len_feat, self.len_feat, dropout=0.3)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feat, out_channels=self.num_classes, kernel_size=1,
            padding=0)
        )

        self.scale_down = nn.Conv1d(in_channels=(self.temp_scale[0]+self.temp_scale[1]+self.temp_scale[2]), out_channels=100, kernel_size=1,
            padding=0)

        # self.classifier_2 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.len_feat, out_channels=self.num_classes, kernel_size=1,
        #     padding=0)
        # )

        # self.classifier_4 = nn.Sequential(
        #     nn.Conv1d(in_channels=self.len_feat, out_channels=self.num_classes, kernel_size=1,
        #     padding=0)
        # )

        self.global_mask = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1),
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=self.temporal_scale, kernel_size=1,stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        # self.global_mask_2 = nn.Sequential(
        #     nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1),
        #     nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=256, out_channels=self.temp_scale[1], kernel_size=1,stride=1, padding=0, bias=False),
        #     nn.Sigmoid()
        # )

        # self.global_mask_4 = nn.Sequential(
        #     nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,padding=1),
        #     # nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3,padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(in_channels=512, out_channels=self.temp_scale[2], kernel_size=1,stride=1, padding=0, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, snip, snip_2, snip_4):

        ### Snippet Embedding Module ###

        ## for temporal length = 100 
        snip = snip.permute(0,2,1)
        out = self.embedding(snip,snip,snip)
        out = out.permute(0,2,1)
        features = out

        ### for temporal length = 200
        snip_2 = snip_2.permute(0,2,1)
        out_2 = self.embedding(snip_2,snip_2,snip_2)
        out_2 = out_2.permute(0,2,1)
        features_2 = out_2

        ### for temporal length = 400
        snip_4 = snip_4.permute(0,2,1)
        out_4 = self.embedding(snip_4,snip_4,snip_4)
        out_4 = out_4.permute(0,2,1)
        features_4 = out_4

        feat_all = torch.cat((out,out_2,out_4), 2)
        # print(feat_all.size())
        feat_all = feat_all.permute(0,2,1)
        feat_all = self.scale_down(feat_all).permute(0,2,1)

        # print(feat_all.size())

        ### Classifier Branch ###
        top_br = self.classifier(feat_all)
        # top_br_2 = self.classifier_2(features_2)
        # top_br_4 = self.classifier_4(features_4)
        ### Global Segmentation Mask Branch ###
        bottom_br = self.global_mask(feat_all)
        # bottom_br_2 = self.global_mask_2(features_2)
        # bottom_br_4 = self.global_mask_4(features_4)

        return top_br, bottom_br  





