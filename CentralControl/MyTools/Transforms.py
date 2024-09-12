# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:19:37 2023

@author: ab19109
画像を学習用に加工する
"""
import numpy as np
import math as m
import cv2
import glob
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import random

'''
Randam Erasing Augmentation
***
Zhun Zhong et.al. "Random Erasing Data Augmentation"
arXiv:1708.04896v2
***
'''        
class RandomErase(object):
    def __init__(
            self,
            p = 0.5,
            sl = 0.02,
            sh = 0.4,
            r1 = 0.3,
        ):
        
        #REAを実行する確率(0.0 ~ 1.0)
        self.probability = p
        
        #元画像に対する目隠し領域の面積比の下限と上限
        self.sl = sl
        self.sh = sh
        
        #目隠し領域の縦横比の下限(上限はr1の逆数)
        self.r1 = r1
        
        
    def __call__(self, img):
        # =============================================================================
        # img(tensor): 目隠しを付与する画像．
        # =============================================================================
        
        #乱数を生成し，REA実行確率より小さい値になったらREA実行
        if random.uniform(0, 1) < self.probability:
            #元画像のチャンネル数，高さ，幅
            c, h, w = img.size()
            #元画像の面積
            area = h * w
            
            #目隠し領域の大きさ・縦横比を決める
            while True:
                #目隠し領域の面積
                #元画像に対する目隠し領域の面積比の下限~上限の乱数 * 元画像の面積
                rea_area = random.uniform(self.sl, self.sh) * area
                
                #目隠し領域の縦横比
                #下限~上限の乱数
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
                
                #目隠し領域の高さと幅
                he = int(m.sqrt(rea_area * aspect_ratio))
                we = int(m.sqrt(rea_area / aspect_ratio))
                
                
                #目隠し領域の高さと幅が元画像より小さくなったらループ脱出
                if we < w and he < h:
                    break
                
            #目隠し領域の位置を決める
            #左上のx,y座標
            x1 = random.randint(0, w - we)
            y1 = random.randint(0, h - he)
            
            #目隠し領域をランダムな値でぬりつぶす            
            mask = torch.tensor([random.random() for _ in range(he*we*c)])
            mask = mask.reshape(c, he, we)

            img[:, y1: y1+he, x1: x1+we] = mask
            
        
        return img
        
    
    
'''
画像のランダムな切り出し
始めに画像の縦横を拡大し，その後元画像と同じ大きさの領域を切り出す
'''    
class RandomCrop(object):
    def __init__(
            self,
            p,
            interpolation = Image.BILINEAR,
            expansion = 1.125
        ):
        
        #RandomCrop実行確率
        self.probability = p

        #画像補間の方法
        self.interpolation = interpolation
        
        #画像の縦横の拡大比率
        self.expansion = expansion

        
    def __call__(self, img):
        # =============================================================================
        # img(tensor): 処理する画像        
        # =============================================================================
        
        #画像のチャンネル数，高さ，幅
        c, h, w = img.size()
        to_pil = ToPILImage()
        to_tensor = ToTensor()
        
        #Tensorを一旦PILにする
        pil_img = to_pil(img) 
        
        #乱数を生成し，実行確率を下回る値ならRandomCrop実行
        if random.uniform(0, 1) < self.probability:
            #拡大後の画像の高さ・幅
            new_height = int(h* self.expansion)
            new_width = int(w * self.expansion)
            
            #画像拡大
            resized_img_pil = pil_img.resize((new_width, new_height), self.interpolation)
            
            #切り取る領域の左上x,y座標の最大値(1番右下)
            x_range = new_width - w
            y_range = new_height - h

            #切り取る領域のx,y座標
            x1 = int(random.uniform(0, x_range))
            y1 = int(random.uniform(0, y_range))
            
            #PIL -> Tensor
            resized_img = to_tensor(resized_img_pil)

            #画像切り取り
            croped_img = resized_img[:, y1: y1+h, x1: x1+w]
    
    
        else:
            croped_img = img
            
        
        return croped_img
    
        