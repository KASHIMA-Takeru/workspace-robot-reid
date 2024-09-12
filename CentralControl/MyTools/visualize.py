# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:54:59 2024

@author: ab19109
主成分分析でCNNの学習の様子を可視化したい
学習データとテストデータから抽出した特徴ベクトルを2次元上にプロットする
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
import os.path as osp

import cv2
import glob
from natsort import natsorted

import matplotlib.pyplot as plt


'''
特徴ベクトルの主成分分析を行い，2次元に削減・プロットする
学習とテストで場合分けする
学習→ミニバッチごとにプロットし，リアルタイム表示・保存
=> リアルタイム表示は没
テスト→指定されたエポックごとにプロットし，保存のみ
エポックごとに行い，プロット結果を保存，あとで動画にする
'''
def plot_features(features, labels, num_classes, save_dir, mode, epoch, batch_idx=None, len_batch=None):
    # =============================================================================
    # features (list[tensor or ndarray]): 抽出された特徴ベクトルのリスト
    # labels (list[int]): データの正解ラベルのリスト．特徴ベクトルのリストと対応している
    # num_classes (int): 分類するクラス数
    # save_dir(str): グラフの保存先    
    # mode (str): 学習かテストか
    # epoch (int): 現在のエポック数．
    # batch_idx (int): 現在のミニバッチ更新回数.学習時のみに使う
    # len_batch(int): ミニバッチ更新回数の合計．学習時のみに使う
    # =============================================================================
    
    if mode != 'train' and mode != 'test':
        raise ValueError("'{}' is unsuitable mode. mode must be either 'train' or 'test'".format(mode))
        
    '''
    主成分分析
    '''
    features = np.concatenate(features, 0)
    labels = np.concatenate(labels, 0)

    #データの標準化
    scaler = StandardScaler()
    f_scaled = scaler.fit_transform(features)
    #PCA実行．2次元平面上にプロットするため，主成分を2つまで取得
    pca = PCA(n_components = 2)
    f_pca = pca.fit_transform(f_scaled)

    #プロットするデータ
    x_data = f_pca[:, 0]
    y_data = f_pca[:, 1]
# =============================================================================
#     features[labels==label_idx, 0],
#     features[labels==label_idx, 1],
# =============================================================================
    
    '''
    結果プロット
    '''   
    #凡例作成
    legends = sorted(set(labels))
    #クラスごとにプロットする色を指定
    
    #plt.ion()
    plt.clf()

    #特徴ベクトルごとにプロット
    for label_idx in legends:
        plt.scatter(
            x_data[labels==label_idx],
            y_data[labels==label_idx],
            c = 'C{}'.format(label_idx),
            s = 5
            )
        
    plt.legend(legends, loc='upper right')
    plt.title("Epoch: {}".format(epoch), loc='left')
    
    #plt.draw()
    #plt.pause(0.001)
    
    save_name = 'epoch_{}.png'.format(epoch)
    


    '''
    保存
    '''
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(osp.join(save_dir, save_name))

    plt.close()    
        


'''
プロット結果を動画にする
'''
def make_video(file_dir, title):
    # =============================================================================
    # file_dir (str): 動画作成に使う画像があるフォルダのディレクトリ
    # title (str): 動画のタイトル．現状CNNの改良点をそのままタイトルにする予定    
    # =============================================================================
    
    #画像取得
    #natsortedでファイルを数字順に並び替える
    image_data = natsorted(glob.glob(osp.join(file_dir, '*.png')))
    
    print("Making video from plot images...")
    print("Got {} images from {}".format(len(image_data), file_dir))

    
    #画像サイズ, FPS
    image_size = (640, 480)
    fps = 10
    
    #動画作成
    fmt = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(osp.join(file_dir, title + '.mp4'), fmt, fps, image_size, isColor=True)
    
    for image in image_data:
        img = cv2.imread(image)
        img = cv2.resize(img, image_size)
        writer.write(img)
        
    writer.release()

