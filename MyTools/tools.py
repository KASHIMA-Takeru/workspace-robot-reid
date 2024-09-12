# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:38:13 2024

@author: ab19109
細かい処理をまとめて書くところ
"""
import numpy as np
import cv2
import os
import os.path as osp
import glob
from PIL import Image
import json
import re



'''
画像パスからIDを取得する
'''
def get_id(path):
    # =============================================================================
    # path (str): 画像のパス    
    # =============================================================================

    #ファイル名を_で区切り，先頭を個人ID，2番目をカメラIDとする
    return osp.basename(path).split('_')[:2]


'''
Re-IDのRank-k精度を計算する
'''
def calc_rankk(query_set, all_rank_list, k):
    # =============================================================================
    # query_set (list): 入力画像のパスが入ったリスト
    # all_rank_list (list): 入力画像ごとのランクイン画像のリストが入ったリスト
    # k (int): 計算する順位     
    # =============================================================================

    #全入力画像における正解率を入れていくリスト
    AP = []
    #入力画像ごとのループ
    for qpath, rank_list in zip(query_set, all_rank_list):
        #正解数
        n_correct = 0
        
        qpid, _ = get_id(qpath)
        for rpath in rank_list[:k]:
            rpid, _ = get_id(rpath[0])
            
            if qpid == rpid:
                n_correct += 1
                
        AP.append(n_correct / k)
        
    rank_k = np.mean(AP)
    
    return rank_k*100


'''
2枚の画像が同一人物かを判定する
'''
def check_match(input1, input2):
    # =============================================================================
    # input1, input2 (str): 画像パス    
    # =============================================================================

    #両者の個人IDを取得
    qpid, _ = get_id(input1)
    gpid, _ = get_id(input2)
    
    return qpid == gpid



'''
Re-IDの結果画像を作成する
→入力画像と同一人物と判断された画像を並べ，正解・不正解により外枠の色を分ける
'''
def visrank(query, rank_list, topk):
    # =============================================================================
    # query (str): 入力画像のパス
    # rank_list(list): [入力画像と同一人物と判断された画像のパス, その画像を囲う色]が入ったリスト
    # =============================================================================
    
    #画像サイズ
    width = 128
    height = 256
    
    
    num_cols = topk + 1
    #順位を表す変数
    rank = 1
    
    GRID_SPACING = 10
    QUERY_EXTRA_SPACING = 90
    BW = 5
    
    #入力画像読み込み+リサイズ
    qimg = cv2.resize(cv2.imread(query), (width, height))
    
    #黒枠で囲う
    qimg = cv2.copyMakeBorder(
        qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0,0,0)
        )
    #外枠の太さ分，画像サイズが変わったため再びリサイズ
    qimg = cv2.resize(qimg, (width, height))
    
    #Re-IDの結果が入る所
    grid_img = 255 * np.ones(
        (
            height,
            num_cols*width + topk*GRID_SPACING + QUERY_EXTRA_SPACING, 3
        ),
            dtype = np.uint8
    )
    grid_img[:, :width, :] = qimg

    for r, c in rank_list:
        rimg = cv2.resize(cv2.imread(r), (width, height))
        rimg = cv2.copyMakeBorder(
            rimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=c
            )
        rimg = cv2.resize(rimg, (width, height))
        start = rank*width + rank*GRID_SPACING + QUERY_EXTRA_SPACING
        end = (
            rank + 1
            ) * width + rank*GRID_SPACING + QUERY_EXTRA_SPACING
        
        grid_img[:, start:end, :] = rimg
        
        rank += 1
        
    
    return grid_img



'''
Re-IDの結果画像の作成
'''
def visrank2(query, rank_list, color_list, topk):
    # =============================================================================
    # query (str): 入力画像のパス
    # rank_list(list): [入力画像と同一人物と判断された画像のパス, その画像を囲う色]が入ったリスト
    # =============================================================================
    
    #画像サイズ
    width = 128
    height = 256
    
    
    num_cols = topk + 1
    #順位を表す変数
    rank = 1
    
    GRID_SPACING = 10
    QUERY_EXTRA_SPACING = 90
    BW = 5
    
    #入力画像読み込み+リサイズ
    qimg = cv2.resize(cv2.imread(query), (width, height))
    
    #黒枠で囲う
    qimg = cv2.copyMakeBorder(
        qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0,0,0)
        )
    #外枠の太さ分，画像サイズが変わったため再びリサイズ
    qimg = cv2.resize(qimg, (width, height))
    
    #Re-IDの結果が入る所
    grid_img = 255 * np.ones(
        (
            height,
            num_cols*width + topk*GRID_SPACING + QUERY_EXTRA_SPACING, 3
        ),
            dtype = np.uint8
    )
    grid_img[:, :width, :] = qimg

    for r, c in zip(rank_list, color_list):
        rimg = cv2.resize(cv2.imread(r), (width, height))
        rimg = cv2.copyMakeBorder(
            rimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=c
            )
        rimg = cv2.resize(rimg, (width, height))
        start = rank*width + rank*GRID_SPACING + QUERY_EXTRA_SPACING
        end = (
            rank + 1
            ) * width + rank*GRID_SPACING + QUERY_EXTRA_SPACING
        
        grid_img[:, start:end, :] = rimg
        
        rank += 1
        
    
    return grid_img


'''
AP計算
'''
def calc_ap(raw_cmc):
    '''
    Parameters
    ----------
    raw_cmc : ndarray / list
        Re-IDの結果の正誤判定が入った配列．1/0で表されている

    Returns
    -------
    ap: float
        APの値

    '''
    #print("inputed raw cmc > ", raw_cmc)
    raw_cmc = raw_cmc.flatten()
    #print("flatten >", raw_cmc)
    if isinstance(raw_cmc, list):
        #print("cmc > ", raw_cmc)
        raw_cmc = np.ndarray(raw_cmc)
        #print("raw_cmc > ", raw_cmc)
        
    #予測結果が全て不正解の場合，APがNaNになるので，0を返すようにする
    if raw_cmc.all() == 0:
        #print("return 0.0")
        
        AP =  0.0
    
    else:
        #print("raw cmc > ", raw_cmc)
        cmc = raw_cmc.cumsum()
        #print("cmc > ", cmc)
        
        cmc[cmc > 1] = 1
        #print("cmc (after)> ", cmc)
        
        num_rel = raw_cmc.sum()
        #print("num rel > ", num_rel)
        tmp_cmc = raw_cmc.cumsum()
        #print("tmp cmc > ", tmp_cmc)
        
        tmp_cmc = [x / (i+1.0) for i, x in enumerate(tmp_cmc)]
        #print("tmp cmc > ", tmp_cmc)
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        #print("tmp cmc > ", tmp_cmc)
        
        AP = tmp_cmc.sum() / num_rel
        #print("AP > ", AP)
    
    return AP

