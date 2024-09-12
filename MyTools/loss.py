# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:03:38 2024

@author: ab19109
CNNの学習に使う損失関数を設計する
"""
import torch
import torch.nn as nn
from torch.autograd.function import Function


'''
Cross Entropy Loss
Label Smoothingも適用する
Torchreidにもあるけどこっちに書いておいた方が楽そう
'''
class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, eps=0.1, use_gpu=True):
        super(CrossEntropyLoss, self).__init__()
        
        #分類クラス数
        self.num_classes = num_classes
        #Label Smoothingに用いる重み
        self.eps = eps
        #GPUを使うかどうか
        self.use_gpu = use_gpu

        #PytorchのLogSoftmax関数
        self.logsoftmax = nn.LogSoftmax(dim=1)
        

    def forward(self, inputs, targets):
        # =============================================================================
        # inputs (tensor): Softmaxに通す前の予測結果. (batch_size, num_classes) の形
        # targets (tensor): 各データの正解ラベル        
        # =============================================================================
        
        #予測スコア算出
        log_probs = self.logsoftmax(inputs)
        
        #log_probsと同じサイズの0埋めtensor
        zeros = torch.zeros(log_probs.size())
        targets = zeros.scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        
        if self.use_gpu:
            targets = targets.cuda()
            log_probs = log_probs.cuda()
        
        else:
            targets = targets.cpu()
            log_probs = log_probs.cpu()
            
        targets = (1 - self.eps) * targets + self.eps / self.num_classes
        
        loss = (-targets * log_probs).mean(0).sum()
        
        return loss



'''
Center Loss
***
A Discriminative Feature Learning Approach for Deep Face Recognition
***
'''
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim=512, use_gpu=False):
        super(CenterLoss, self).__init__()
        
        #分類クラス数
        self.num_classes = num_classes
        #特徴ベクトルの次元数
        self.feat_dim = feature_dim
        
        #GPU使用か
        self.use_gpu = use_gpu
        
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim)).cuda()
        
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

            
            
    def forward(self, x, labels):
        # =============================================================================
        # x (tensor): バッチ内データの特徴ベクトルたち
        # labels (tensor): 各データの正解ラベル        
        # =============================================================================
        
        #print("centers > ", self.centers.device)
        batch_size = x.size(0)
        
        if self.use_gpu:
            x = x.cuda()
        else:
            x = x.cpu()
        
        #print("x > ", x.shape)
        #print("centers > ", self.centers.shape)
        #ある特徴ベクトルとそのクラスの特徴ベクトルたちの中心の距離
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)          
        classes = torch.arange(self.num_classes).long()
        #classes = tensor([0, 1, 2, ...])
          
        if self.use_gpu:
            classes = classes.cuda()
        else:
            classes = classes.cpu()
        
        #バッチデータの正解ラベルを縦に並べた形にする
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        labels = labels.cpu()
        #正解ラベルに対応する箇所のみTrue，他の場所はFalseになる
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        
        #mask.float() -> Trueなら1.0, Falseなら0.0になる
        dist = distmat * mask.float()
        #dist.clamp() -> distの要素のうち最小値と最大値を指定した値に置き換える
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        
        return loss



'''
Center Lossの別ver.
'''
class CenterLoss2(nn.Module):
    def __init__(self, num_classes, feature_dim, use_gpu=False):
        super(CenterLoss2, self).__init__()
        
        if use_gpu:
            self.centers = nn.Parameter(torch.randn(num_classes, feature_dim)).cuda()
        
        else:
            self.centers = nn.Parameter(torch.randn(num_classes, feature_dim)).cpu()
            
        self.centerlossfunc = CenterLossFunc.apply
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu
        
        
    def forward(self, features, labels):
        # =============================================================================
        # features (tensor): 特徴ベクトル(batch_size x feature_dim)
        # label (tensor): 正解ラべル (batch_size)        
        # =============================================================================
        
        if self.use_gpu:
            features.cuda()
            labels.cuda()
        
        else:
            features = features.cpu()
            labels = labels.cpu()
        
        
        #print("labels > ", labels)
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        
        batch_size_tensor = features.new_empty(1).fill_(1)
        #print("batch_size_tensor > ", batch_size_tensor)        

        
        loss = self.centerlossfunc(features, labels, self.centers, batch_size_tensor)
        
        return loss
        

class CenterLossFunc(Function):
    @staticmethod
    def forward(ctx, features, labels, centers, batch_size):
        ctx.save_for_backward(features, labels, centers, batch_size)
        centers_batch = centers.index_select(0, labels.long())
        
        dist = (features - centers_batch).pow(2).sum() / 2.0 / batch_size
        
        
        return dist
    
    @staticmethod
    def backward(ctx, grad_output):
        features, labels, centers, batch_size = ctx.saved_tensors
        #print("ctx >", ctx.saved_tensors)
        #print("batch size > ", batch_size)
        #中心座標のうちバッチ内データに含まれるIDの座標が入っている．順番はバッチ内データに対応．
        centers_batch = centers.index_select(0, labels.long())
        
        #中心座標と特徴ベクトルのずれ
        diff = centers_batch - features
        
        #分類クラス数分の1.0埋めテンソル(-> 各IDに対応している)
        counts = centers.new_ones(centers.size(0))
        #バッチサイズ分の1.0埋めテンソル
        ones = centers.new_ones(labels.size(0))
        #[分類クラス数，特徴次元]のサイズを持った0.0埋めテンソル 
        grad_centers = centers.new_zeros(centers.size())
        
        
        #バッチ内データに含まれる各IDの数．初期値が1なので実際に含まれるデータ数より1多い値が入っている
        counts = counts.scatter_add_(0, labels.long(), ones)
        
        #バッチ内データのIDを縦に並べる→特徴ベクトルと同じ次元に拡大→型変換
        grad_centers.scatter_add_(0, labels.unsqueeze(1).expand(features.size()).long(), diff)
        #.view(-1, 1)でcountsを縦並びのテンソルにする
        grad_centers = grad_centers / counts.view(-1, 1)
        
        
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None        
        

