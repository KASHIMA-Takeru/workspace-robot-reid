# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:40:44 2024

@author: ab19109
2つのベクトルのユークリッド距離やを計算する
"""
import torch
import numpy as np


def calc_euclidean_dist(input1, input2):
    # =============================================================================
    # input1, input2 (tensor): 距離を計算したい2つのベクトル    
    # =============================================================================

    '''
    計算前の確認
    '''
    #入力がテンソルか確認
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    
    #入力が2次元か確認
    assert input1.dim() == 2, \
        'input1: Expected 2-D tensor, but got {}-D tensor'.format(input1.dim())
    
    assert input2.dim() == 2, \
        'input2: Expected 2-D tensor, but got {}-D tensor'.format(input2.dim())
        
    #2つのベクトルのサイズが同じか確認
    assert input1.size(1) == input2.size(1), \
        'Both input must be the same size ({} and {})'.format(input1.size(), input2.size())
        
    
    '''
    距離計算
    '''
    m = input1.size(0)
    n = input2.size(0)
    
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    distmat = mat1 + mat2
    
    distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
    
    
    return distmat



def calc_euclidean_dist2(input1, input2):
    '''
    Parameters
    ----------
    input1 : Tensor
        入力画像の特徴ベクトル
    input2 : Tensor
        検索画像の特徴ベクトル

    Returns
    -------
    2つのベクトル間のユークリッド距離(float)

    '''
    
    input1_np = np.array(input1.cpu())
    input2_np = np.array(input2.cpu())
    
    dist = np.linalg.norm(input1_np - input2_np)
    #-> 上の関数と同じ結果になった
    
    
    return dist
    

    
    

