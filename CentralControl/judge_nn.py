# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:16:02 2024

@author: ab19109
身体部位ごとの特徴ベクトル間の距離から2枚の画像が同一人物であるかを判定するNNを作りたい
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import csv
import numpy as np
import random as rd


'''
判定を行うNNクラス
'''
class JudgeNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        #入力層
        self.N_IN = 11
        
        #中間層
        self.N_MID1 = 15
        self.N_MID2 = 20
        self.N_MID3 = 20
        self.N_MID4 = 15
        self.N_MID5 = 10
        self.N_MID6 = 8
        self.N_MID7 = 5
        self.N_MID8 = 3
    
        #出力層
        self.N_OUT = 2
        
        
        #FC層
        self.fc1 = nn.Linear(self.N_IN, self.N_MID1)
        self.fc2 = nn.Linear(self.N_MID1, self.N_MID2)
        self.fc3 = nn.Linear(self.N_MID2, self.N_MID3)
        self.fc4 = nn.Linear(self.N_MID3, self.N_MID4)
        self.fc5 = nn.Linear(self.N_MID4, self.N_MID5)
        self.fc6 = nn.Linear(self.N_MID5, self.N_MID6)
        self.fc7 = nn.Linear(self.N_MID6, self.N_MID7)
        self.fc8 = nn.Linear(self.N_MID7, self.N_MID8)
        self.fc9 = nn.Linear(self.N_MID8, self.N_OUT)
        
        
    
    '''
    順伝搬
    '''  
    def forward(self, x):
        y1 = torch.relu(self.fc1(x))
        y2 = torch.relu(self.fc2(y1))
        y3 = torch.relu(self.fc3(y2))
        y4 = torch.relu(self.fc4(y3))
        y5 = torch.relu(self.fc5(y4))
        y6 = torch.relu(self.fc6(y5))
        y7 = torch.relu(self.fc7(y6))
        y8 = torch.relu(self.fc8(y7))
        y9 = self.fc9(y8)
        
        return y9
            
    
'''
NN ver2
'''
class JudgeNN2(nn.Module):
    def __init__(self):
        super().__init__()
        
        #入力層
        self.N_IN = 11
        
        #中間層
        self.N_MID = 5
    
        #出力層
        self.N_OUT = 2
        
        
        #FC層
        self.fcin = nn.Linear(self.N_IN, self.N_MID)
        self.fcmid = nn.Linear(self.N_MID, self.N_MID)
        self.fcout = nn.Linear(self.N_MID, self.N_OUT)

    def forward(self, x):
        y = torch.relu(self.fcin(x))
        for i in range(2):
            y = torch.relu(self.fcmid(y))

        y = self.fcout(y)

        return y
        

'''
NN ver3
'''
class JudgeNN3(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 3, 1)
        self.conv2 = nn.Conv1d(3, 3, 1)
        self.conv3 = nn.Conv1d(3, 1, 1)
        self.conv4 = nn.Conv1d(1, 1, 1)
        
        self.fc = nn.Linear(11, 2)
    
    def multi_conv(self, x, n_conv):
        for i in range(n_conv):
            x = self.conv2(x)
            
        return x
    
    
    def forward(self, x):
        y1 = torch.relu(self.conv1(x))

        y2a = self.multi_conv(y1, 1)
        y2b = self.multi_conv(y1, 2)
        y2c = self.multi_conv(y1, 3)
        #print("y1 = ", y1)
        y2 = torch.relu(y2a+y2b+y2c)
        y3 = torch.relu(self.conv3(y2))
        y4 = x + y3
        y5 = torch.relu(self.conv4(y4))
        #print("y4 = ", y4)
        y6 = self.fc(y5)
        
        return y6
        

'''
Attention ver
'''
class JudgeAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(11, 11)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(11, 2)
        
        self.conv1 = nn.Conv1d(1, 1, 2)
        
        self.bn = nn.BatchNorm1d(1)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        
        x_skip = self.fc3(x)
        y1 = torch.sigmoid(self.fc1(x))
        y2 = torch.sigmoid(self.fc1(y1))
        y3 = torch.sigmoid(self.fc1(y2))
        y4 = torch.sigmoid(self.fc1(y3))
        
       
        out = self.fc3(y4) + x_skip

        return out    
        

'''
単純なFC層ver
'''    
class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(11, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 2)
        
    def forward(self, x):
        y1 = torch.relu(self.fc1(x))
        y2 = torch.relu(self.fc2(y1))
        y3 = torch.relu(self.fc3(y2))
        y = self.fc4(y3)
        
        return y
        
    

'''
学習に使うデータセットのクラス
'''
class JudgeData():
    def __init__(self):
        
        self.e_mess = 'Data has not been loaded yet'
        self.n_train_sample = self.e_mess
        self.n_valid_sample = self.e_mess
        self.n_test_sample = self.e_mess
        
        
    def __getlen__(self):
        
        return self.n_train_sample, self.n_valid_sample

    #データを読み込む関数
    def load_data(self, csv_path, mode, shuffle):
        '''
        Parameters
        ----------
        csv_path : str
            データが書かれたcsvファイルのパス.
            csvの各行rowは以下の構成になっていることが前提
            row[:2]: ペア画像のファイル名
            row[2:13]: ペア画像の各身体部位画像の特徴ベクトル間の距離
            row[13]: ペア画像が同一人物か異なる人物かのラベル
            
        mode : str
            読み込むデータが学習データか検証データか
            
        shuffle : bool
            データをシャッフルするか
            
        Returns
        -------
        got_data : str
            {
                'samples': ペア画像のファイル名のリスト,
                'datas': 身体部位画像の特徴ベクトル間の距離の配列
                'lebels': 同一人物 or 異なる人物のラベルの配列
                }

        '''
        
        #データを入れる辞書
        got_data = {
            'samples': None,
            'datas': None,
            'lebals': None
            }
        
        assert mode == 'train' or mode == 'valid' or mode == 'test', print("'mode' must be either 'train', 'valid', or 'test' . Got '{}'".format(mode))
        
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            data_file = csv.reader(csv_file)
            
            #ペア画像のファイル名，データ(特徴ベクトル間の距離)，ラベルを入れていくリスト
            samples = []
            datas = []
            labels = []
            
            for i, data in enumerate(data_file):
                samples.append(data[:2])
                datas.append(data[2:13])
                labels.append(data[13])
        
        #データ数
        n_sample = i + 1
        if mode == 'train':
            self.n_train_sample = n_sample
            
        elif mode == 'valid':
            self.n_valid_sample = n_sample
            
        elif mode == 'test':
            self.n_test_sample = n_sample
        
        #データのシャッフル
        if shuffle:
            #要素番号をシャッフル
            indices = list(range(n_sample))
            rd.shuffle(indices)
            
            samples_s = []
            datas_s = []
            labels_s = []
            for j in indices:
                samples_s.append(samples[j])
                datas_s.append(datas[j])
                labels_s.append(labels[j])
                
            datas_s_array = np.asarray(datas_s, dtype=np.float64)
            labels_s_array = np.asarray(labels_s, dtype=np.int64)
            
            got_data['samples'] = samples_s
            got_data['datas'] = datas_s_array
            got_data['labels'] = labels_s_array
            
        else:
            #リストを配列に変換
            datas_array = np.asarray(datas, dtype=np.float64)
            labels_array = np.asarray(labels, dtype=np.int64)
            
            got_data['samples'] = samples
            got_data['datas'] = datas_array
            got_data['labels'] = labels_array
        
        return got_data
        
            
            
            
            
            
                    

