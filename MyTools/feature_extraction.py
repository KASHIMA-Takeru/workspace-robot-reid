# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:33:36 2023

@author: ab19109
画像をCNNに入力して特徴ベクトルを出力する関数
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image



class MyFeatureExtractor(object):
    def __init__(self, model, image_size):
        
        #パラメータ類
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        device = 'cuda'
        
        
        #Transform function
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        
        preprocess = T.Compose(transforms)
        
        to_pil = T.ToPILImage()
        
        device = torch.device(device)
        model.to(device)
        
        #Class attributes
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.to_pil = to_pil
        
        
    def __call__(self, input):
        if isinstance(input, list):
            images = []

            for element in input:
                if isinstance(element, str):
                    image = Image.open(element).convert('RGB')

                elif isinstance(element, np.ndarray):
                    image = self.to_pil(element)

                else:
                    raise TypeError(
                        'Type of each element must belong to [str | numpy.ndarray]'
                    )

                image = self.preprocess(image)
                images.append(image)
            images = torch.stack(images, dim=0)
            images = images.to(self.device)

        elif isinstance(input, str):
            image = Image.open(input).convert('RGB')
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)
            
        elif isinstance(input, np.ndarray):
            
            image = self.to_pil(input)
            image = self.preprocess(image)
            images = image.unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            images = input.to(self.device)

        else:
            raise NotImplementedError

        with torch.no_grad():
            
            features = self.model(images)

        return features
        




def feature_extractor(model, image, image_size):
    '''
    Parameters
    ----------
    model : nn.module
        特徴抽出に用いるCNN
    image : ndarray / list / str / Tensor
        画像
    image_size : tuple
        画像サイズ(H x W)

    Returns
    -------
    features : Tensor
        画像の特徴ベクトル
    '''
    #print("Feature Extractor")
    
    '''
    パラメータ類
    '''
    pixel_mean = [0.485, 0.456, 0.406]
    pixel_std = [0.229, 0.224, 0.225]
    pixel_norm = True
    device = 'cuda'
    #print("fe#1")
    
    #transform関数の作成    
    transforms = []
    transforms += [T.Resize(image_size)]
    transforms += [T.ToTensor()]  
    #print("fe#2")
    if pixel_norm:
        transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
    
    preprocess = T.Compose(transforms)
    
    to_pil = T.ToPILImage()
    
    device = torch.device(device)
    model.to(device)
    #print("fe#3")
    
    '''
    前準備
    '''
    if isinstance(image, list):
        #print("fe#4")
        images = []
        print("image > ", image)
        for element in image:
            if isinstance(element, str):
                #print("fe#4.1")
                image = Image.open(element).convert('RGB')
                
            elif isinstance(element, np.ndarray):
                #print("fe#4.2")
                image = to_pil(element)
                #print("fe#4.3")
            else:
                raise TypeError(
                    "Type of each element must be belong to [str | numpy.ndarray]"
                )
                
            image = preprocess(image)
            images.append(image)
            
        images = torch.stack(images, dim=0)
        images = images.to(device)
        
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')
        image = preprocess(image)
        images = image.unsqueeze(0).to(device)
        
    elif isinstance(image, np.ndarray):
        #print("fe#5")
        image = to_pil(image)
        image = preprocess(image)
        images = image.unsqueeze(0).to(device)
        
    elif isinstance(image, torch.Tensor):
        input_image = Image.open(image)
        if input_image.dim() == 3:
            image = image.unsqueeze(0)
        images = image.to(device)
        
    else:
        raise NotImplementedError

    '''
    特徴抽出
    '''
    #print("fe#6")
    with torch.no_grad():
        features = model(images)
        #print("fe#7")

    return features
