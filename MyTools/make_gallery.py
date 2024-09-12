# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:09:17 2024

@author: ab19109
検索データベースを作成する
"""
from openpose_processor import make_part_image, make_person_image

import numpy as np
import cv2
import os
import os.path as osp

import datetime


'''
カメラ画像から人物を検出し，人物画像と身体部位画像を作成して保存する関数
'''
def make_gallery(image, keypoints, save_dir):
    '''
    Parameters
    ----------
    image : np.ndarray
        カメラ画像
    keypoints : list
        人物のキーポイントの座標をまとめたリスト
    save_dir: str
        画像保存先

    Returns
    -------
    None.

    '''
    
    #作成した画像のファイル名を入れていくリスト
    name_list = []

    #画像ファイルに振り分ける番号．
    i = 1
    
    #人物領域のバウンディングボックスの座標取得
    bbox, _ = make_person_image(image, key_list=keypoints)
    #身体部位画像作成
    part_images = make_part_image(image, keypoints)
    
    #人物画像保存
    #画像の名前に使う時刻
    timestamp = datetime.datetime.now().strftime("%H%M_%S")
    
    #保存名の決定.かぶりがなくなるまでループする
    while True:
        name = timestamp + '_{}'.format(i)
        
        if name in name_list:
            i += 1
            continue
        
        else:
            i = 1
            name_list.append(name)
            
            break
        
    #人物画像
    print("bbox > ", bbox)
    person = image[bbox[0]: bbox[1], bbox[2], bbox[3]]
    cv2.imwrite(osp.join(save_dir, name + '.jpg'), person)
    
    #部位画像の保存
    for part in part_images.keys():
        part_save_folder = osp.join(save_dir, 'part', part)
        os.makedirs(part_save_folder, exist_ok=True)
        
        part_save_name = name + '_{}.jpg'.format(part)
        cv2.imwrite(osp.join(part_save_folder, part_save_name), part_images[part])
    
    
    
    
    
    
    

