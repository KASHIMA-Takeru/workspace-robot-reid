# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:07:04 2024

@author: ab19109
OpenPoseを使った処理を行う関数
"""
import os
import sys
import numpy as np
import cv2



'''
OpenPoseの準備
'''
#実行ファイルのパス
dir_path = r'C:\Users\ab19109\.spyder-py3'


sys.path.append(dir_path + r'\openpose\build\python\openpose\Release');
os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + r'\openpose\build\x64\Release;' + \
    dir_path + r'\.spyder-py3\openpose\build\bin;'

import pyopenpose as op

#Custom Params
params = dict()
params["model_folder"] = r'C:\Users\ab19109\.spyder-py3\openpose\models'

#Start OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()


'''
人物画像からキーポイントの位置を検出する関数
複数人が写っていた場合は，その人数分のキーポイントが返される
'''
def openpose_key(image, disp=False):
    # =============================================================================
    # image (ndarray): キーポイントを検出したい画像    
    # =============================================================================
    
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    
    #キーポイントの検出結果(座標をまとめたリストと線で結んだ画像)
    keypoints = datum.poseKeypoints
    keyimage = datum.cvOutputData
    
    if disp:
        cv2.namedWindow("Input image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("keyimage", cv2.WINDOW_NORMAL)
        cv2.imshow("Input image", image)
        cv2.imshow("keyimage", keyimage)
        cv2.waitKey(0)
    
    return keypoints, keyimage


'''
カメラ画像から人物領域の画像を作成する関数
複数人が写っている場合は，それぞれの画像を作成する．
'''
def make_person_image(image, key_list=None, thrs=0.1, ex_len=100):
    # =============================================================================
    # image (ndarray): カメラ画像
    # key_list (list): キーポイントの座標のリスト (予め検出しておいた場合はここに入る．必須ではない)
    # thrs (float): OpenPoseで検出されたキーポイントの信頼度の閾値．キーポイントの信頼度がこの値未満の場合，
    #               そのキーポイントは検出されていないものとする．
    # ex_len(int): 人物領域を決めるときの追加余白[mm] 
    # =============================================================================

    #画像の高さと幅
    h, w, _ = image.shape
    
    #キーポイントの座標取得
    if key_list == None:
        key_list, keyimage = openpose_key(image)

    
    
    #人物領域の端を決める
    bbox_list = []
    #人物画像を入れていくリスト
    people_list = []
    '''
    指標
    ・左右・下端は検出されたキーポイントのうち最も端にある部分+追加ピクセル分
    ・上端は顔や後頭部の画像の作成時と同じで，耳や目の位置から顔の長さを概算する
    ・追加ピクセルは，身体部位の画像を作ったときと同じで，キーポイント間のピクセル数と人体寸法データベースの値から決める
    ・基本は顔~足のピクセル数から求めるが，足が写っていないときは膝や腰の位置を使う
    '''
    #キーポイントのリスト内のループ．person: 1人のキーポイントのリスト
    for person in key_list: 
        
        '''
        切り取り時の追加ピクセル計算
        全身が収まるようにしたあとにさらに追加分を足す
        '''
        #信頼度が0だとキーポイントの座標も0になるので，信頼度が閾値以下のキーポイントを除外
        key = np.where(person[:, 2] > thrs)
        #key (tuple): 信頼度が閾値以上であったキーポイントの要素番号とdtypeが入っている
        
        #検出されたキーポイントの数が少ない場合は，人物画像は作らず，Re-IDも行わない
        if len(key[0]) < 10:
            continue
        
        '''
        上端の追加ピクセル
        →耳か目が一番上のキーポイントの想定
        ・顔の上端の位置を求める
        ・顔の上端から更に追加分を足す
        '''
        #両耳の位置が推定されている場合
        if person[17][2] > thrs and person[18][2] > thrs:
            #pm: 両耳間の距離(A3: 145.7)から算出
            pm = abs(person[17][0] - person[18][0]) / 145.7
            
            #顔の上端．両耳のうち上にある方~頭頂部まで(A33: 135.4)
            face_top = max(0, min(person[17][1], person[18][1]) - pm * 135.4)
            
            
            #上端の追加ピクセル．耳の位置 - 顔の上端
            ex_top = min(person[17][1], person[18][1]) - face_top
            #print("#1")
            
        #片耳のみ位置が推定されている場合
        #右耳は推定されているが左耳は推定されていない場合
        elif person[17][2] > thrs and person[18][2] < thrs:
            #pm: 鼻~頭部後端(A21: 199.5)と耳~頭部後端(A27: 87.4)から算出
            pm = abs(person[0][0] - person[17][0]) / (199.5 - 87.4)
            
            #顔の上端．耳~頭頂部
            face_top = max(0, person[17][1] - pm * 135.4)
            
            #追加ピクセル
            ex_top = person[17][1] - face_top
            #print("#2")
            
        #左耳の位置は推定されているが右耳の位置は推定されていない場合       
        elif person[18][2] > thrs and person[17][2] < thrs:
            #pm: 鼻~頭部後端(A21: 199.5)と耳~頭部後端(A27: 87.4)から算出
            pm = abs(person[0][0] - person[18][0]) / (199.5 - 87.4)
            
            #顔の上端．耳~頭頂部
            face_top = max(0, person[18][1] - pm * 135.4)
            
            #追加ピクセル
            ex_top = person[18][1] - face_top + pm * ex_len
            
            #print("#3")
            
        #print("Face top > ", face_top)
        #print("Extra top > ", ex_top)
        
        '''
        人物領域の決定
        '''
        #上下左右端
        top = max(0, round(min(person[key][:, 1]) - ex_top))
        bottom = min(h, round(max(person[key][:, 1]) + pm * ex_len))
        left = max(0, round(min(person[key][:, 0]) - pm * ex_len))
        right = min(w, round(max(person[key][:, 0]) + pm * ex_len))
        
        #人物領域の面積が小さかったら人物領域は作成しない
        #基準：64 x 128
        if (bottom - top) < 128  or (right - left) < 64:
            continue
        
        else:
            bbox_list.append([top, bottom, left, right])
            #人物領域に沿って画像切り抜き
            person = image[top: bottom, left: right]
            people_list.append(person)
        
    return people_list, key_list, keyimage
    
    
    

    
    