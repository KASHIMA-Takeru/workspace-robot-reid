# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:27:11 2024

@author: ab19109
Re-IDを行うクラスの雛型を作りたい．今後ロボットに搭載できるように書いていきたい
→過去に作成したクラス(reid.py，reid2.py)は時間計測とか閾値のループとかが入っていて他で使えないので．
"""
import torch
import torch.nn as nn
import os
import os.path as osp
import glob
import tqdm

import numpy as np
import shutil
import cv2
import time

import sys
import csv

import MyTools as myt
from MyTools import openpose_processor as opp

from mymodels import ModelDict
import mymodels as mym


'''
Re-IDを行うクラス
'''
class ReIDBase:
    
    def __init__(self, pivod_dict: dict, save_dir: str,
                 use_part: bool = True, thrs: float = 20.0, maxk: int = 50):
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
    def prepare(self, gpath, part_path):
        '''
        Parameters
        ----------
        gpath : str
            検索画像があるフォルダのパス
        
        ppath: str
            身体部位画像があるフォルダのパス
        Returns
        -------
        None.

        '''
    
        '''
        CNNの準備
        '''
        #全身画像を学習したCNNの準備
        model_wholebody = mym.build_model(
            name = self.pivod_dict['wholebody']['model_name'],
            )
        model_wholebody = model_wholebody.cuda()
        model_wholebody.eval()
        
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
            #画像ファイル名取得
            gname = osp.splitext(osp.basename(gimg))[0]
            self.gname_list.append(gname)
            #画像パスからID取得
            gid, _ = myt.get_id(gimg)
            self.gid_list.append(gid)
            
            #CNNに画像を入力して特徴ベクトル抽出
            gf = myt.feature_extractor(self.pivod_dict['wholebody']['model'], gimg, self.pivod_dict['wholebody']['size'])
            #辞書に追加            
            self.gf_list.append(gf.clone().detach())
            
            #身体部位画像の特徴抽出
            if self.use_part:
                self.gallery_part_data[gname] = {k: None for k in self.pivod_dict.keys()}
                #身体部位ごとのループ
                for part in self.part_list:
                    #身体部位画像のパス
                    ppath = osp.join(part_path, part, gname + '_{}.jpg'.format(part))
                    #print("part image path > ", ppath)
                    #身体部位の画像が存在したらCNNに入力して特徴抽出
                    if osp.isfile(ppath):
                        gpf = myt.feature_extractor(self.pivod_dict[part]['model'], ppath, self.pivod_dict[part]['size'])
                        #辞書に追加
                        self.gallery_part_data[gname][part] = gpf
        
        self.gf_list = torch.cat(self.gf_list, dim=0)
        print("Finish")
        #print("Gallery IDs > ", self.gid_list)
        
    
    '''
    Re-IDを行う関数
    '''
    def run_reid(self, people, frame, target, keypoints, video_mode=False):
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
        video_mode: bool
            →ロボットを使う時はFalse，事前に撮影した動画で行うときはTrueにしておく

        Returns
        -------
        target_person: str
            入力人物の中のどこに追尾対象がいるか．人物検出の段階で検出された人物に仮IDを付与しておき，
            それとtarget_personを照らし合わせることで，画像中のどこに追尾対象がいるかを判断する予定

        '''
        
        #全身画像をCNNに入力して特徴ベクトル抽出
        qfs = myt.feature_extractor(self.pivod_dict['wholebody']['model'], people, self.pivod_dict['wholebody']['size'])
        
        #特徴ベクトル間の距離計算
        distmat = myt.calc_euclidean_dist(qfs, self.gf_list).cpu()
     
        #print("distmat > ", distmat)
        
        #ベクトル間の距離の順位
        indices = np.argsort(distmat.cpu(), axis=1)
        indices = np.asarray(indices)
        #print("indices > ", indices)
        #距離が最短の画像の位置
        min_pos = [np.argmin(dist).item() for dist in distmat]
        #print("min position > ", min_pos)
        
        #距離が最短だった検索データのID取得
        cids = [self.gid_list[j] for j in min_pos]
        print("candidate IDs > ", cids)
        print("indices > ", indices)
        #print("type > ", type(indices))
        #print("shape > ", indices.shape)
        #print("0 > ", indices[0])
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
                print("======")
        
                    
                #cv2.imshow("Query", qimg)
                #cv2.waitKey(1)
                #print("key > ", key)
                #入力人物の仮ID
                temp_id = str(qidx).zfill(3)
                temp_id_list.append(temp_id)
                q_part_images = opp.make_part_image(frame, key)
                self.query_data[temp_id] = q_part_images
                #入力画像と特徴ベクトル間の距離が短い候補画像の位置が入ったリスト
                index_list = indices[qidx]
                #print("index list > ", index_list)
                #全ての候補画像の計算結果をいれていくリスト
                all_dist_list = []
                #print("pivod dict > ", self.pivod_dict.keys())
                #print("query data > ", self.query_data[temp_id].keys())
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
                    print("Re-ID with part images")
                    for part in self.part_list:
                        #print("--- {} ---".format(part))
                        try:
                            
                            #cv2.imshow("query {}".format(part), self.query_data[temp_id][part])
                            #cv2.waitKey(0)
                            #print("Candidate {} > ".format(part), type(self.gallery_part_data[cname][part]))
                            #身体部位画像の特徴抽出
                            #print("query part image: ", type(self.query_data[temp_id][part]))
                            qpf = myt.feature_extractor(self.pivod_dict[part]['model'], self.query_data[temp_id][part], self.pivod_dict[part]['size'])
                            #print("#1")
                            #候補画像の身体部位の特徴
                            cpf = self.gallery_part_data[cname][part]
                            #print(part, cpf)
                            #print("#2")
                            #print("Candidate part feature > ", type(cpf))
                            
                            #ベクトル間の距離計算
                            pdist = myt.calc_euclidean_dist(qpf, cpf)
                            #print("#3")
                            #計算結果を辞書に追加
                            pd_dict[part] = self.pivod_dict[part]['weight'] * pdist.item()
                            #print("distance > ", pdist.item())
                        
                        except NotImplementedError:
                            #print("Not implemented error")
                            #print("{} image does not exist".format(part))
                            #print("query {} > ".format(part), type(self.query_data[temp_id][part]))
                            pd_dict[part] = None
                            
                        except ValueError:
                            #print("Value error")
                            #print("query {} image: ".format(part), type(self.query_data[temp_id][part]))
                            #print("shape: ", self.query_data[temp_id][part].shape)
                            #print("candidate {} image: ".format(part), type(cpf))
                            
                            pd_dict[part] = None
                            #入力画像の部位が正常に切り取られなかったら特徴抽出の段階でここに来る
                            #→画像の幅か高さが0になる場合がある
                            #OpenPoseへの入力が人物画像だったのが原因ぽい
                            
                        except AssertionError:
                            #print("Assertion error")
                            #print("input for calc_euclidean_dist is not correct")
                            #print("candidate {} image: ".format(part), type(cpf))
                            #候補画像の部位画像が存在しない場合は，ベクトル間の距離計算の段階でここに来る
                            pd_dict[part] = None
                            
                        except KeyError:
                            #print("Key error")
                            #print("{} image of query doesn't exist!".format(part))
                            #print("keys: ", self.query_data[temp_id].keys())
                            pd_dict[part] = None
                            #入力画像の部位画像が存在しない場合は特徴抽出の段階でここに来る
  
                    #加重平均計算
                    values = list(pd_dict.values())
                    print("values > ", values)
                    values = [v for v in values if v != None]
                    
                    factor = np.mean(values)
                    #pprint.pprint(pd_dict)
                    #print("Factor >  {:.3f}".format(factor))
                    all_dist_list.append(factor)
                    
                    #閾値より小さかったら同一人物
                    if self.thrs > factor:
                        print("{} was Detected as {}".format(temp_id, cid))
                        pid_list.append(cid)
                        factor_list.append(factor)
                        break
                    
                    else:
                        
                        factor_list.append(factor)
                        
                    #探索する最大人数を超えた場合
                    if i > self.maxk:
                        print("Could not detect person")
                        #ベクトル間の距離の加重平均値が最小の人物のIDを入力人物のIDとする
                        min_index = all_dist_list.index(min(all_dist_list))
                        print("min index: ", min_index)
                        cid = self.gid_list[min_index]
                        print("cid: ", cid)
                        pid_list.append(cid)
                        
                        break
                    
                #cv2.destroyWindow("Query")
                
            print("=== Finish Re-ID ===")
            print("target ID > ", target)
            print("temp ID > ", temp_id_list)
            print("ID list > ", pid_list)
            print("factor list > ", factor_list)
            #入力人物の中に追尾対象がいるが，複数人が追尾対象と判断された場合
            if ((target in pid_list) and (pid_list.count(target) > 1)) and not video_mode:
                #追尾対象のIDがリストのどこにあるか
                duplicate_index = [i for i, x in enumerate(pid_list) if x==target]
                print("duplicate index > ", duplicate_index)
                #追尾対象と同一人物と判断された人たちのfactorの値
                factors = [factor_list[j] for j in duplicate_index]
                print("factors > ", factors)
                #ベクトル間の距離の加重平均が最小の人を同一人物とする
                truth_index = factors.index(min(factors))
                target_person = temp_id_list[duplicate_index[truth_index]]
                
                
            #入力人物の中に追尾対象がいて，対象と判断された人が1人のみだった場合
            elif (target in pid_list) and (pid_list.count(target) == 1):
                target_index = pid_list.index(target)
                target_person = temp_id_list[target_index]
                
            #入力人物の中に追尾対象がいなかった場合
            elif target not in pid_list:
                target_person = -1
                    
            
            return target_person, pid_list
        
        

'''
ReIDBaseの改良版
'''
class ReID():
    def __init__(self, save_dir: str, gpath: str, gppath: str, reid_freq: int, use_part: bool, 
                 use_nn: bool, thrs: float=20.0, maxk: int=20, save_img: bool=False):
        
        self.reid_Freq = reid_freq
        #CNN，加重平均に使う重み，画像サイズをまとめた辞書
        self.pivod_dict = dict()
        #身体部位のリスト
        self.part_list = []
        
        #検索画像と身体部位画像の保存先
        self.gpath = gpath
        self.gppath = gppath
        
        #保存先
        self.save_dir = save_dir
        #身体部位画像を使ったRe-IDを行うか
        self.use_part = use_part
        #身体部位画像でのRe-IDに判定NNを使うか
        self.use_nn = use_nn,
        #身体部位画像でのRe-IDでに使う閾値
        self.thrs = thrs,
        #身体部位画像でのRe-IDで探索する最大人数
        self.maxk = maxk
        #Re-IDに使われた画像を保存するか
        self.save_img = save_img
        #画像の保存先
        self.save_folder = r''
        
        #検索画像を入れていくリスト
        self.gallery_images = []
        #検索データの個人IDを入れていくリスト
        self.gid_list = []
        #検索データのファイル名を入れていくリスト
        self.gname_list = []
        #検索データの全身画像の特徴ベクトルを入れるリスト
        self.gf_list = []
        #検索データの身体部位画像の特徴ベクトルを入れる辞書．IDをキーとし，身体部位画像の特徴ベクトルを値とする．
        self.gallery_part_data = dict()
        
        #同一人物か判定するNN
        self.mynn = None
        #Softmax関数
        self.softmax = nn.Softmax(dim=0)
        
        #結果記録用のcsv, txt
        self.f = None
        self.writer = None
        
        self.logf = None


    '''
    準備
    '''
    def prepare(self, mynn=None, nn_path='', id_list=[]):
        '''
        Parameters
        ----------

        mynn: torch.nn
            判定を行うNNモデル
        nn_path : str
            判定NNのモデルのパス
        id_list : list
            識別する人物のIDが入ったリスト
            
        '''
        
        #結果記録用のcsv, txt
        self.f = open(osp.join(self.save_dir, 'log.csv'), "a", newline="")
        self.writer = csv.writer(self.f)
        
        self.logf = open(osp.join(self.save_dir, 'memo.txt'), "w")
        
        #CNNをまとめた辞書の準備
        mydict = ModelDict()
        if self.use_part:
            self.pivod_dict = mydict.get_item(separate=False)
        
        else:
            self.pivod_dict['wholebody'], _ = mydict.get_item(separate=True)
        
        self.part_list = list(self.pivod_dict.keys())
        
        if 'wholebody' in self.part_list:
            self.part_list.remove('wholebody')
        
        if self.save_img:
            self.image_save_path = osp.join(self.save_dir, 'reid_images')
            os.makedirs(self.image_save_path, exist_ok=True)
        
        '''
        検索データの特徴抽出
        '''       
        self.gallery_images = glob.glob(osp.join(self.gpath, '*.jpg'))
        print("Extracting features from gallrey set with {} images...".format(len(self.gallery_images)))
        for gimg, _ in zip(self.gallery_images, tqdm.tqdm(range(len(self.gallery_images)))):
            #画像ファイル名取得
            gname = osp.splitext(osp.basename(gimg))[0]
            #画像ファイル名からID取得
            gid, _ = myt.get_id(gname)

            if gid in id_list:
                self.gname_list.append(gname)
                self.gid_list.append(gid)
                #画像をCNNに入力して特徴ベクトル抽出
                gf = myt.feature_extractor(self.pivod_dict['wholebody']['model'], gimg, self.pivod_dict['wholebody']['size'])
                #リストに追加
                self.gf_list.append(gf.clone().detach())
                
                #身体部位画像の特徴抽出
                if self.use_part:
                    self.gallery_part_data[gname] = {k: None for k in self.part_list}
                    #身体部位ごとのループ
                    for part in self.part_list:
                        #身体部位画像のパス
                        ppath = osp.join(self.gppath, part, gname + '_{}.jpg'.format(part))
                        
                        #身体部位画像のパスが存在したらCNNに入力して特徴抽出
                        if osp.isfile(ppath):
                            gpf = myt.feature_extractor(self.pivod_dict[part]['model'], ppath, self.pivod_dict[part]['size'])
                            #辞書に追加
                            self.gallery_part_data[gname][part] = gpf.clone().detach()
                        
        self.gf_list = torch.cat(self.gf_list, dim = 0)
        
        #判定NNの読み込み
        if self.use_nn == True:
            print("Load pretrained weights")
            mynn.load_state_dict(torch.load(nn_path))
            mynn.cuda()
            mynn.eval()
            
            self.mynn = mynn
        print("Ready for Re-ID")
        
        self.writer.writerow(['Frame', 'person', 'time'])
        
        print("=== Basic Information ===", file=self.logf)
        mark = 'o' if self.use_part == True else 'x'
        method = 'NN' if self.use_nn == True else 'weighted average' 
        
        print(" - Use body part images: ", mark, file=self.logf)
        if self.use_part:
            print(" - Decision method: ", method, file=self.logf)
            
        if self.use_nn == True:
            print(" - NN: ", self.mynn, file=self.logf)
            
        else:
            print(" - Threshold: ", self.thrs, file=self.logf)
            print(" - Weight", file=self.logf)
            
            for part in self.part_list:
                print("    Part: ", part, self.pivod_dict[part]['weight'], file=self.logf)
            
        print(" - Maximum number for exploring: ", self.maxk, file=self.logf)
            
        
        
    '''
    Re-IDを行う関数
    '''
    def run_reid(self, people, frame, keypoints, n_frame, tracking=False, target=None, patch=-100):
        '''
        Parameters
        ----------
        people : list
            人物画像のリスト．入力画像として扱う．
        frame : np.ndarray
            動画のフレーム
        keypoints : np.ndarray
            OpenPoseで検出した人物のキーポイントの配列
        n_frame : int
            動画のフレーム数
        tracking : bool
            特定人物を追尾するか
        target : str
            追尾する人物のID
        patch : int
            身体部位画像を使ったRe-IDで，身体部位の特徴ベクトル間の距離が計算できないときに代わりに入れる値．
            the default is -100

        Returns
        -------
        pid_list : list
            Re-IDで推定された各人物のIDのリスト
        '''
        
        assert patch < 0, "'patch' must be negative value. (now patch = {})".format(patch)
        
        if tracking and target == None:
            raise NotImplementedError("'tracking' is True but 'target' is not set")
        
        try:
            #処理開始時刻
            start = time.perf_counter()        

            '''
            全身画像でのRe-ID
            '''
            print("#1")
            #全身画像をCNNに入力して特徴ベクトル抽出
            qfs = myt.feature_extractor(self.pivod_dict['wholebody']['model'], people, self.pivod_dict['wholebody']['size'])            
            #検索画像の特徴ベクトルとの距離を計算
            distmat = myt.calc_euclidean_dist(qfs, self.gf_list).cpu()
        
            #ベクトル間の距離の順位
            #各画像ごとに，特徴ベクトル間の距離が短い検索画像の要素番号が順に入る
            indices = np.asarray(np.argsort(distmat.cpu(), axis=1))
            #print("indides: ", indices)
                
            #距離が最短の検索画像の位置
            min_pos = [np.argmin(dist).item() for dist in distmat]
            #距離が最短だった検索画像のID取得(全身画像でのRe-IDの結果)
            pid_list = [self.gid_list[j] for j in min_pos]
            #print("min pos: ", min_pos)
            #print("pid list: ", pid_list)
            print("#2")
        
            '''
            身体部位画像でのRe-ID
            '''
            if self.use_part:
                print("#3")
                #同一人物かの演算結果を入れていくリスト
                factor_list = []
                #入力人物の仮IDをいれていくリスト
                temp_id_list = []
            
                #入力画像ごとのループ　
                for qidx, (qimg, key) in enumerate(zip(people, keypoints)):
                    print("#4")
                    print("Re-ID: {} / {} person".format(qidx+1, len(people)), file=self.logf)
                    #入力人物の身体部位画像作成
                    q_part_images = opp.make_part_image(frame, key)
                
                    #入力画像と特徴ベクトル間の距離が短い候補画像の位置が入ったリスト
                    index_list = indices[qidx]
                    #全ての候補画像の計算結果を入れていくリスト
                    all_factor_list = []
                
                    #入力人物の仮ID
                    temp_id = str(qidx).zfill(3)
                    temp_id_list.append(temp_id)
                
                    #入力画像を保存
                    if self.save_img:
                        save_folder = osp.join(self.image_save_path, 'reid_{}'.format(n_frame), 'query_{}'.format(qidx+1))
                        os.makedirs(save_folder, exist_ok=True)
                        cv2.imwrite(osp.join(save_folder, '_query.jpg'), qimg)
                
                    #候補画像ごとのループ
                    for i, cidx in enumerate(index_list):
                        print("#5")
                        #候補画像のファイル名とID
                        cname = self.gname_list[cidx]
                        print(" - candidate {}: ".format(i+1), cname, file=self.logf)
                        cid, _ = myt.get_id(self.gname_list[cidx])
                    
                        if self.save_img:
                            shutil.copy(self.gallery_images[cidx], osp.join(save_folder, 'candidate_{}.jpg'.format(i+1)))
                        #身体部位ごとの特徴ベクトル間の距離をいれていく辞書
                        #加重平均と閾値を用いる場合は重みとの積を入れる
                        pd_dict = {}
                    
                        #身体部位ごとのループ
                        for part in self.part_list:
                            print("#6")
                        
                            try:
                                #入力人物の身体部位画像の特徴抽出
                                qpf = myt.feature_extractor(self.pivod_dict[part]['model'], q_part_images[part], self.pivod_dict[part]['size'])
                                #候補画像の身体部位画像の特徴
                                cpf = self.gallery_part_data[cname][part]
                                #print("candidate file: ", osp.join(self.gppath, cname+'_{}.jpg'.format(part)))
                                if self.save_img:
                                    #入力・候補画像の保存
                                    partimg_savepath = osp.join(save_folder, 'part', part)
                                    os.makedirs(partimg_savepath, exist_ok=True)
                                    cv2.imwrite(osp.join(partimg_savepath, '_q_{}.jpg'.format(part)), q_part_images[part])
                                    shutil.copy(osp.join(self.gppath, part,  cname+'_{}.jpg'.format(part)), osp.join(partimg_savepath, 'c{}_{}.jpg'.format(i+1, part)))
                                
                                #特徴ベクトル間の距離計算
                                pdist = myt.calc_euclidean_dist(qpf, cpf)
                                #計算結果をリストに追加
                                if self.use_nn:
                                    pd_dict[part] = pdist.item()
                            
                                else:
                                    pd_dict[part] = pdist.item()*self.pivod_dict[part]['weight']
                            #身体部位画像が無い場合
                            except (TypeError, FileNotFoundError, KeyError, NotImplementedError, AssertionError):
                                pd_dict[part] = patch
                        
                            except Exception as e:
                                #身体部位画像が無い場合
                                pd_dict[part] = patch
                                _, _, exception_traceback = sys.exc_info()
                                line_no = exception_traceback.tb_lineno
                                print("Line No.: {}, ".format(line_no), e)
                            
                            
                        #同一人物かの判断
                        if self.use_nn == True:
                            #NNへ入力するデータ
                            values = torch.Tensor(list(pd_dict.values())).cuda()
                            #print("input data for NN: ", values)
                            #NNへ入力．Softmax関数で同一人物である確率を算出する
                            #print("{} / {}".format(i+1, self.maxk))
                            #print("Input: ", values)
                            #print("Candidate: ", cid)
                            y = self.mynn.forward(values)
                            #print("y = ", y)
                            preds = torch.argmax(y)
                            #print("prediction: ", preds)
                            #print("y = ", y)
                            output = self.softmax(y.detach())
                            #print("output: ", output)
                            factor = output[0]
                            #print("factor: ", factor)
                            #同一人物である確率が0.5以上なら同一人物と判断
                            judge = (preds == 0)
                            print(" - output: ", y, file=self.logf)
                        
                 
                        else:
                            print("#7")
                            #加重平均計算
                            values = [v for v in pd_dict.values() if v != patch]
                            factor = np.mean(values)
                            all_factor_list.append(factor)
                            print(" - factor: {:.3f}".format(factor), file=self.logf)
                        
                            #加重平均が閾値未満なら同一人物
                            judge = self.thrs > factor
                        
                        factor_list.append(factor)
                        

                        #同一人物と判断された場合
                        if judge:
                            print("#8")
                            #全身画像でのRe-IDで予測されたIDを置き換え
                            pid_list[qidx] = cid

                            break
                    
                        else:
                            print("#9")
                            all_factor_list.append(factor)
                    
                    
                        #探索する最大人数に達しても入力人物を識別できなかった場合
                        if i+1 >= self.maxk:
                            print("#10")
                            if self.use_nn:
                                #同一人物の可能性が一番高かった人物と同一人物とする
                                id_index = all_factor_list.index(max(all_factor_list))
                        
                            else:
                                print("#11")
                                #特徴ベクトル間の距離の加重平均が最小の人物と同一人物とする
                                id_index = all_factor_list.index(min(all_factor_list))
                            
                            cid = self.gid_list[id_index]
                            pid_list[qidx] = cid
                        
                            print("Could not find the match person", file=self.logf)

                            break
        
            eta = time.perf_counter() - start
            #csvに書き込む行
            row = [n_frame, len(people), eta]
            self.writer.writerow(row)
        
            #特定人物を追尾する場合の処理
            if tracking:
                print("#13")
                #入力人物の中に追尾対象がいるが，複数人が追尾対象と判断された場合
                if ((target in pid_list) and (pid_list.count(target) > 1)):
                    #追尾対象のIDがリストのどこにあるか
                    duplicate_index = [i for i, x in enumerate(pid_list) if x==target]
                    print("duplicate index > ", duplicate_index)
                    #追尾対象と同一人物と判断された人たちのfactorの値
                    factors = [factor_list[j] for j in duplicate_index]
                    print("factors > ", factors)
                    #ベクトル間の距離の加重平均が最小の人を同一人物とする
                    truth_index = factors.index(min(factors))
                    target_person = temp_id_list[duplicate_index[truth_index]]
                
                
                #入力人物の中に追尾対象がいて，対象と判断された人が1人のみだった場合
                elif (target in pid_list) and (pid_list.count(target) == 1):
                    target_index = pid_list.index(target)
                    target_person = temp_id_list[target_index]
                
                #入力人物の中に追尾対象がいなかった場合
                elif target not in pid_list:
                    target_person = -1
                    
            
                return target_person, pid_list
        
            else:
            
                return pid_list

        except Exception as e:
            print(e)



    def release(self):
        self.f.close()
        self.logf.close()
            
            
            
            
            
            
            
            
            
        
        
        
        
        