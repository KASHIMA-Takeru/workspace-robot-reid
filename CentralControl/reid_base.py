# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:27:11 2024

@author: ab19109
Re-IDを行うクラスの雛型を作りたい．今後ロボットに搭載できるように書いていきたい
→過去に作成したクラス(reid.py，reid2.py)は時間計測とか閾値のループとかが入っていて他で使えないので．
"""
from multiprocessing import Value
import sys
import os
import os.path as osp
import glob
import tqdm

import torch
import torch.nn as nn
import numpy as np
import cv2

import datetime

import pprint
import itertools

import MyTools as myt
from MyTools import openpose_processor as opp
import mymodels as mym
from judge_nn import MyNN


'''
Re-IDを行うクラス
'''
class ReIDBase:
    
    def __init__(self, pivod_dict: dict, save_dir: str,
                 use_part: bool = True, use_nn: bool = True, thrs: float = 20.0, maxk: int = 50):
        '''
        Parameters
        ----------
     
        Example:
            dict_wholebody = {
                'path': 'model.pth.tar-60',
                'model_name': 'osnet',
                'size': (256, 128) #(height, width)
                }
            
        pivod_dict : dict
            全身画像を学習したモデルのパス，CNNの名前，CNNに入力する画像のサイズ,
            身体部位の画像を学習したモデルのパス，CNNの名前，CNNに入力する画像のサイズ，加重平均に用いる重みが入った辞書

        Example:
            pivod_dict = {
                'whole_body': {
                    'path': 'model.pth.tar-60',
                    'model_name':  'osnet',
                    'size': (2556, 128) #(height, width)
                    }
                'face': {
                    'weight': 2.0,
                    'path': 'model.pth.tar-60',
                    'model_name': 'osnet',
                    'size': (256, 128) #(height, width)
                    },
                'back_head': {
                    ...
                    } 
                ...
                'left_foot': {
                    ...
                    }
                }
        
        save_dir: str
            結果の保存先
        use_part : bool, optional
            身体部位画像を使ったRe-IDを行うか． The default is True.
        thrs : float, optional
            身体部位画像を使ったRe-IDで同一人物か異なる人物かを判断する閾値．The default is 20.0.
        maxk : int, optional
            身体部位画像を使ったRe-IDで検索データを探索する最大人数. The default is 50.

        Returns
        -------
        None.

        '''

        self.pivod_dict = pivod_dict
        #身体部位のリスト
        self.part_list = list(pivod_dict.keys())
        self.part_list.remove('wholebody')
        
        self.save_dir = save_dir
        self.use_part = use_part
        self.thrs = thrs
        self.maxk = maxk
        self.use_nn = use_nn
        self.mynn = MyNN()
        self.softmax = nn.Softmax(dim=0)
 

        #入力画像のデータを入れる辞書
        #複数人物が検出されたときに，全身画像でのRe-IDは一括で行いたい．
        #キーを検出された番号にして，値を更に辞書にして，キーを全身画像，各部位にしてその値を画像(配列)または特徴ベクトルにする?
        #→画像を入れた方が扱いやすい?
        self.query_data = {}
        
        #検索データの個人IDを入れていくリスト
        self.gid_list = []
        #検索データのファイル名を入れていくリスト
        self.gname_list = []
        #検索データの全身画像の特徴ベクトルを入れるリスト
        self.gf_list = []
        #検索データの身体部位画像の特徴ベクトルを入れる辞書．IDをキーとし，身体部位画像の特徴ベクトルを値とする．
        self.gallery_part_data = {'name': {k: None for k in pivod_dict.keys()}}
        #self.gallery_data = {}
        
        
    '''
    モデルの読み込み，検索画像の特徴抽出を行う関数
    '''
    def prepare(self, gpath, nn_path: str, id_list: list):
        '''
        Parameters
        ----------
        gpath : str
            検索画像があるフォルダのパス
            
        Returns
        -------
        None.

        '''
    
        '''
        CNNの準備
        '''
        try:
            #全身画像を学習したCNNの準備
            model_wholebody = mym.build_model(
                name = self.pivod_dict['wholebody']['model_name'],
                pretrained = False
                )
            model_wholebody = model_wholebody.cuda()
            model_wholebody.eval()
            print("built model")
            mym.load_model(model_wholebody, self.pivod_dict['wholebody']['path'])
            self.pivod_dict['wholebody']['model'] = model_wholebody        
        
            if self.use_part:
                #身体部位画像を学習したCNNの準備
                for part in self.part_list:
                    #CNN準備
                    model = mym.build_model(
                        name = self.pivod_dict[part]['model_name']
                        )
                    model = model.cuda()
                    model.eval()
                    #学習済みファイル読み込み
                    mym.load_model(model, self.pivod_dict[part]['path'])
                    self.pivod_dict[part]['model'] = model
        
            if self.use_nn:
                self.mynn.load_state_dict(torch.load(nn_path))
                self.mynn.cuda()
                self.mynn.eval()

            '''
            Summary表示
            '''
            print("="*3 + " Summary " + "="*100)
            print("Part" + " "*8 + "|CNN" + " "*28 + "|Model" + " "*40 + "|Size" + " "*8 + "|Load")
            for part in self.pivod_dict.keys():
                name = self.pivod_dict[part]['model_name']
                model = osp.basename(self.pivod_dict[part]['path'])
                height = self.pivod_dict[part]['size'][0]
                width = self.pivod_dict[part]['size'][1]
                size = "{} x {}".format(height, width)
                load = "O" if osp.isfile(self.pivod_dict[part]['path']) else "X"
            
                print(part + " "*(13-len(part)) + name + " "*(32-len(name)) + model + " "*(46-len(model)) + size + " "*(12-len(size)), load)
            print("="*112)
        
            '''
            検索データの特徴抽出
            '''
            print("Extracting gallery features...")
            #画像取得
            gallery_images = glob.glob(osp.join(gpath, '*.jpg'))
            for gimg, _ in zip(gallery_images, tqdm.tqdm(range(len(gallery_images)))):
                
                #画像パスからID取得
                gid, _ = myt.get_id(gimg)
                #print(gid)
                if gid in id_list:
                    #print("#2")
                    self.gid_list.append(gid)
                    #画像ファイル名取得
                    gname = osp.splitext(osp.basename(gimg))[0]
                    #print("#1")
                    self.gname_list.append(gname)
                
                    #CNNに画像を入力して特徴ベクトル抽出
                    gf = myt.feature_extractor(self.pivod_dict['wholebody']['model'], gimg, self.pivod_dict['wholebody']['size'])
                    #追加            
                    #print("#3")
                    #print("type: ", type(self.gf_list))
                    self.gf_list.append(gf.clone().detach())
            
                    #身体部位画像の特徴抽出
                    if self.use_part:
                        self.gallery_part_data[gname] = {k: None for k in self.pivod_dict.keys()}
                        #身体部位ごとのループ
                        for part in self.part_list:
                            #身体部位画像のパス
                            ppath = osp.join(gpath, 'part', part, gname + '_{}.jpg'.format(part))
                            #print("part image path > ", ppath)
                            #身体部位の画像が存在したらCNNに入力して特徴抽出
                            if osp.isfile(ppath):
                                gpf = myt.feature_extractor(self.pivod_dict[part]['model'], ppath, self.pivod_dict[part]['size'])
                                #辞書に追加
                                self.gallery_part_data[gname][part] = gpf
        
            self.gf_list = torch.cat(self.gf_list, dim=0)
        
                #print("Gallery IDs > ", self.gid_list)

        except Exception as e:
            
            print(e)
        
    
    '''
    Re-IDを行う関数
    '''
    def run_reid(self, people, frame, target, keypoints, patch=-100):
        '''
        Parameters
        ----------
        people : list
            カメラで検出された人物画像のリスト
        frame : np.ndarray
            カメラ画像
        target: str
            追尾対象のID
        keypoints: list
            人物のキーポイントの座標が入ったリスト

        Returns
        -------
        target_person: str
            入力人物の中のどこに追尾対象がいるか．人物検出の段階で検出された人物に仮IDを付与しておき，
            それとtarget_personを照らし合わせることで，画像中のどこに追尾対象がいるかを判断する予定

        '''
        try:
            #print("Run Re-ID")
            #全身画像をCNNに入力して特徴ベクトル抽出
            qfs = myt.feature_extractor(self.pivod_dict['wholebody']['model'], people, self.pivod_dict['wholebody']['size'])
    
            #特徴ベクトル間の距離計算
            distmat = myt.calc_euclidean_dist(qfs, self.gf_list).cpu()
            #print("distmat > ", distmat)
            
            #ベクトル間の距離の順位
            indices = np.argsort(distmat.cpu(), axis=1)
            indices = np.asarray(indices)
    
            #距離が最短の画像の位置
            min_pos = [np.argmin(dist).item() for dist in distmat]
            #print("min position > ", min_pos)
            
            #距離が最短だった検索データのID取得
            cids = [self.gid_list[j] for j in min_pos]
            #target_person = indices[0]
            print("cids : ", cids)
            if not self.use_part:
                pid_list = cids
                target_flag = False
                target_person = -1
                
                for idx, cid in enumerate(cids):

                    print("cid: ", cid)
                    if str(cid).zfill(3) == target:
                        target_person = str(idx).zfill(3)
                        target_flag = True
                        print("Target person: ", target_person)
                        break
                
                
                                        
            #身体部位画像でのRe-ID
            if self.use_part:
                #入力画像の最終的なIDを入れるリスト
                pid_list = []
                #身体部位画像でのRe-IDの加重平均値をいれていくリスト
                factor_list = []
                #仮IDのリスト
                temp_id_list = []
                #print("ketpoints > ", keypoints)
                
                #入力画像ごとのループ
                for qidx, (qimg, key) in enumerate(zip(people, keypoints)):
    
                    #入力人物の仮ID
                    temp_id = str(qidx).zfill(3)
                    temp_id_list.append(temp_id)
    
                    q_part_images = opp.make_part_image(frame, key)
                    self.query_data[temp_id] = q_part_images
                    #入力画像と特徴ベクトル間の距離が短い候補画像の位置が入ったリスト
                    index_list = indices[qidx]
    
                    #全ての候補画像の計算結果をいれていくリスト
                    all_dist_list = []
                    all_factor_list = []
    
                    #候補画像ごとのループ
                    for i, cidx in enumerate(index_list):
                        #print("Candidate {} / {}".format(i+1, len(index_list)))
                        #print("indices > ", cidx)
                        
                        #候補画像のファイル名
                        cname = self.gname_list[cidx]
                        print("Name: ", cname)
                        #候補画像のID
                        cid = myt.get_id(self.gname_list[cidx])[0]
                        #print("Candidate index >", c_index)
                        #print("Candidate ID> ", self.gid_list[c_index])
                        #print("Candidate name > ", self.gname_list[c_index])                    
                        
                        pd_dict = {}
                        #身体部位ごとのループ
                        #print("Re-ID with part images")
                        for part in self.part_list:
                            #print("--- {} ---".format(part))
                            try:
                                
                                #cv2.imshow("query {}".format(part), self.query_data[temp_id][part])
                                #cv2.waitKey(0)
                                #print("Candidate {} > ".format(part), type(self.gallery_part_data[cname][part]))
                                #身体部位画像の特徴抽出
                                #print("query part image: ", type(self.query_data[temp_id][part]))
                                qpf = myt.feature_extractor(self.pivod_dict[part]['model'], self.query_data[temp_id][part], self.pivod_dict[part]['size'])
                                #候補画像の身体部位の特徴
                                cpf = self.gallery_part_data[cname][part]
                                #print("Candidate part feature > ", type(cpf))
                                
                                #ベクトル間の距離計算
                                pdist = myt.calc_euclidean_dist(qpf, cpf)
    
                                #計算結果を辞書に追加
                                pd_dict[part] = self.pivod_dict[part]['weight'] * pdist.item()
                                #print("distance > ", pdist.item())
                            
                            except (NotImplementedError, ValueError, AssertionError, KeyError):
                                if self.use_nn:
                                    pd_dict[part] = patch
                                else:
                                    pd_dict[part] = None
                        #NNを使った判断
                        if self.use_nn == True:
                            values = torch.tensor(list(pd_dict.values())).cuda()
                            y = self.mynn.forward(values)
                            prediction = torch.argmax(y)
                            output = self.softmax(y.detach())
                            factor = output[0]
                            all_factor_list.append(output)
    
                            #同一人物かの判断
                            if prediction == 0:
                                pid_list.append(cid)
                                factor_list.append(factor)
    
                                break
    
                        else:
                            #加重平均計算
                            values = list(pd_dict.values())
                            #print("values > ", values)
                            values = [v for v in values if v != None]
                        
                            factor = np.mean(values)
                            #pprint.pprint(pd_dict)
                            print("Factor >  {:.3f}".format(factor))
                            all_dist_list.append(factor)
                            
                            #閾値より小さかったら同一人物
                            if self.thrs > factor:
                                #print("{} was Detected as {}".format(temp_id, cid))
                                pid_list.append(cid)
                                factor_list.append(factor)
                                print("find")
                            
                                break
    
                            else:
                                all_dist_list.append(factor)
                                #print("Detected as different person. Factor: ", factor)
                        
    
                        #探索する最大人数を超えた，検索データ全てと照合した場合
                        if i > self.maxk or i > len(index_list):
                            #print("Could not detect person")
                            #ベクトル間の距離の加重平均値が最小の人物のIDを入力人物のIDとする
                            try:
                                min_index = all_dist_list.index(min(all_dist_list))
                            
                                cid = (self.gid_list[min_index])
                                pid_list.append(cid)
                                factor_list.append(factor)
                                print("Could not find target")
                            except:
                                print("Could not find target")
                            break
                        
                    #cv2.destroyWindow("Query")
                    
                #print("=== Finish Re-ID ===")
                #print("target ID > ", target)
                #print("temp ID > ", temp_id_list)
                #print("ID list > ", pid_list)
                #print("factor list > ", factor_list)
                #入力人物の中に追尾対象がいるが，複数人が追尾対象と判断された場合
                if (target in pid_list) and (pid_list.count(target) > 1):
                    #追尾対象のIDがリストのどこにあるか
                    duplicate_index = [i for i, x in enumerate(pid_list) if x==target]
                    #print("duplicate index > ", duplicate_index)
                    #追尾対象と同一人物と判断された人たちのfactorの値
                    factors = [factor_list[j] for j in duplicate_index]
                    #print("factors > ", factors)
                    if self.use_nn:
                        #同一人物の確率が最も高い人を同一人物とする
                        truth_index = factors.index(max(factors))
                    else:
                        #ベクトル間の距離の加重平均が最小の人を同一人物とする
                        truth_index = factors.index(min(factors))
                    target_person = temp_id_list[duplicate_index[truth_index]]
                    target_flag = True
                    
                #入力人物の中に追尾対象がいて，対象と判断された人が1人のみだった場合
                elif (target in pid_list) and (pid_list.count(target) == 1):
                    target_index = pid_list.index(target)
                    target_person = temp_id_list[target_index]
                    target_flag = True
                    
                #入力人物の中に追尾対象がいなかった場合
                elif target not in pid_list:
                    target_person = -1
                    target_flag = False    
                
            return target_person, pid_list, target_flag
        
        except Exception as e:
            print(e)




