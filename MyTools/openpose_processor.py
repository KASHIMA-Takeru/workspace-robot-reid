# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 15:50:21 2024

@author: ab19109
OpenPoseを使った人物画像の処理をまとめる
作成する関数
・キーポイントの検出
・全身画像の作成(複数人)
・身体部位画像の作成(1人分)
"""

import sys
import os
import os.path as osp

import numpy as np
import math as m
import cv2



'''
OpenPose準備
'''
#実行ファイルパス
#cur_path = osp.dirname(osp.realpath(__file__))
#cur_path: C:\Users\ab19109\.spyder-py3\MyTools

#1つ上の階層のパス
dir_path = r'C:\Users\ab19109\.spyder-py3'

#環境変数にOpenPoseのフォルダをパスを追加する
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
画像から人物のキーポイントの位置を検出する関数
'''
def detect_keypoints(image: np.ndarray)-> np.ndarray:
    '''
    Parameters
    ----------
    image : np.ndarray
        カメラ画像

    Returns
    -------
    keypoints : np.ndarray
        キーポイントの座標をまとめた配列
    keyimage : np.ndarray
        キーポイントを線で結んだ画像

    '''
    
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    
    #検出結果(キーポイントの座標をまとめた配列とキーポイントを線で結んだ画像)
    keypoints = datum.poseKeypoints
    keyimage = datum.cvOutputData
    
    return keypoints, keyimage
    


'''
カメラ画像から人物領域の画像を作成する関数
複数人写っている場合はそれぞれの画像を作成する
'''
def make_person_image(image: np.ndarray, keypoints: np.ndarray or None, thrs = 0.1, ex_len = 100):
    '''
    Parameters
    ----------
    image : np.ndarray
        カメラ画像.
    keypoints : np.ndarray or None
        人物のキーポイントの検出結果.

    Returns
    -------
    bbox_list: list
        人物領域の４隅の座標が入ったリスト
        
        
    Example:
        
    bbox_list = make_person_image(image)
    
    for (top, bottom, left, right) in bbox_list:
        person = image[top: bottom, left: right]

    '''
    #画像の高さと幅
    h, w, _ = image.shape
    #人物領域の4隅の座標を入れていくリスト(人物画像を入れると処理が重くなりそう?)
    bbox_list = []
    
    
    #keypointsがNoneならキーポイント検出
    if type(keypoints) == np.ndarray:
        pass
    
    elif keypoints == None:
        keypoints, _ = detect_keypoints(image)
        

    #検出した人物ごとのループ
    for keys in keypoints:
        
        key = np.where(keys[:, 2] > thrs)
        #検出されたキーポイント数が少ない場合は人物画像を作らない
        if len(key[0]) < 10:
            continue
        
        '''
        上端の追加ピクセル
        →耳か目が一番上のキーポイントの想定
        ・顔の上端の位置を求める
        ・顔の上端から更に追加分を足す
        '''
        #両耳の位置が推定されている場合
        if keys[17][2] > thrs and keys[18][2] > thrs:
            #pm: 両耳間の距離(A3: 145.7)から算出
            pm = abs(keys[17][0] - keys[18][0]) / 145.7
            
            #顔の上端．両耳のうち上にある方~頭頂部まで(A33: 135.4)
            face_top = max(0, min(keys[17][1], keys[18][1]) - pm * 135.4)
            
            #上端の追加ピクセル．耳の位置 - 顔の上端
            ex_top = min(keys[17][1], keys[18][1]) - face_top
            #print("#1")
            
        #片耳のみ位置が推定されている場合
        #右耳は推定されているが左耳は推定されていない場合
        elif keys[17][2] > thrs and keys[18][2] < thrs:
            #pm: 鼻~頭部後端(A21: 199.5)と耳~頭部後端(A27: 87.4)から算出
            pm = abs(keys[0][0] - keys[17][0]) / (199.5 - 87.4)
            
            #顔の上端．耳~頭頂部
            face_top = max(0, keys[17][1] - pm * 135.4)
            
            #追加ピクセル
            ex_top = keys[17][1] - face_top
            #print("#2")
            
        #左耳の位置は推定されているが右耳の位置は推定されていない場合       
        elif keys[18][2] > thrs and keys[17][2] < thrs:
            #pm: 鼻~頭部後端(A21: 199.5)と耳~頭部後端(A27: 87.4)から算出
            pm = abs(keys[0][0] - keys[18][0]) / (199.5 - 87.4)
            
            #顔の上端．耳~頭頂部
            face_top = max(0, keys[18][1] - pm * 135.4)
            
            #追加ピクセル
            ex_top = keys[18][1] - face_top + pm * ex_len
            
            #print("#3")
            
        #print("Face top > ", face_top)
        #print("Extra top > ", ex_top)
        
        '''
        人物領域の決定
        '''
        #上下左右端
        top = max(0, round(min(keys[key][:, 1]) - ex_top))
        bottom = min(h, round(max(keys[key][:, 1]) + pm * ex_len))
        left = max(0, round(min(keys[key][:, 0]) - pm * ex_len))
        right = min(w, round(max(keys[key][:, 0]) + pm * ex_len))
        
        #人物領域の面積が小さかったら人物領域は作成しない
        #基準：64 x 128
        if (bottom - top) < 128  or (right - left) < 64:
            continue
        
        else:
            bbox_list.append([top, bottom, left, right])
            
            
    return bbox_list


'''
切り取る座標を決める関数
'''
def decide_coordinates(keys, w, h, beta=20):
    '''
    Parameters
    ----------
    keys : np.ndarray
        キーポイントの検出結果
    w : int
        画像の幅.
    h : int
        画像の高さ.
    beta : int, optional
        切り取りの余白分[mm]. The default is 20.

    Returns
    -------
    coordinates : dict
        身体部位画像が入った辞書

    '''


    '''
    keysから部位の名称と値を紐づける辞書を作成
    '''
    #xy座標，信頼度を表す．辞書のキーに使う
    x = 'x'
    y = 'y'
    r = 'r'
    
    nose = {
        'x': keys[0][0],
        'y': keys[0][1],
        'r': keys[0][2]
        }
    
    heart = {
        'x': keys[1][0],
        'y': keys[1][1],
        'r': keys[1][2]
        }
    
    right_shoulder = {
        'x': keys[2][0],
        'y': keys[2][1],
        'r': keys[2][2]
        }
    
    right_elbow = {
        'x': keys[3][0],
        'y': keys[3][1],
        'r': keys[3][2]
        }
    
    right_wrist = {
        'x': keys[4][0],
        'y': keys[4][1],
        'r': keys[4][2]
        }
    
    left_shoulder = {
        'x': keys[5][0],
        'y': keys[5][1],
        'r': keys[5][2]
        }
    
    left_elbow = {
        'x': keys[6][0],
        'y': keys[6][1],
        'r': keys[6][2]
        }
    
    left_wrist = {
        'x': keys[7][0],
        'y': keys[7][1],
        'r': keys[7][2]
        }
    
    center_waist = {
        'x': keys[8][0],
        'y': keys[8][1],
        'r': keys[8][2]
        }
    
    right_waist = {
        'x': keys[9][0],
        'y': keys[9][1],
        'r': keys[9][2]
        }
    
    right_knee = {
        'x': keys[10][0],
        'y': keys[10][1],
        'r': keys[10][2]
        }
    
    
    right_ankle = {
        'x': keys[11][0],
        'y': keys[11][1],
        'r': keys[11][2]
        }
    
    left_waist = {
        'x': keys[12][0],
        'y': keys[12][1],
        'r': keys[12][2]
        }
    
    left_knee = {
        'x': keys[13][0],
        'y': keys[13][1],
        'r': keys[13][2]
        }
    
    left_ankle = {
        'x': keys[14][0],
        'y': keys[14][1],
        'r': keys[14][2]
        }
    
    right_eye = {
        'x': keys[15][0],
        'y': keys[15][1],
        'r': keys[15][2]
        }
    
    left_eye = {
        'x': keys[16][0],
        'y': keys[16][1],
        'r': keys[16][2]
        }
    
    right_ear = {
        'x': keys[17][0],
        'y': keys[17][1],
        'r': keys[17][2]
        }
    
    left_ear = {
        'x': keys[18][0],
        'y': keys[18][1],
        'r': keys[18][2]
        }
    
    left_toe_in = {
        'x': keys[19][0],
        'y': keys[19][1],
        'r': keys[19][2]
        }
    
    left_toe_out = {
        'x': keys[20][0],
        'y': keys[20][1],
        'r': keys[20][2]
        }
    
    left_heel = {
        'x': keys[21][0],
        'y': keys[21][1],
        'r': keys[21][2]
        }
    
    right_toe_in = {
        'x': keys[22][0],
        'y': keys[22][1],
        'r': keys[22][2]
        }
    
    right_toe_out = {
        'x': keys[23][0],
        'y': keys[23][1],
        'r': keys[23][2]
        }
    
    right_heel = {
        'x': keys[24][0],
        'y': keys[24][1],
        'r': keys[24][2]
        }
        
    
    #左上の座標を(x0, y0)，右下の座標を(x1, y1)とする．
    #部位をまとめた辞書を作成し，座標のリスト[x0, x1, y0, y1]を値にする
    coordinates = {
        'face': None,
        'back_head': None,
        'chest': None,
        'back': None,
        'right_arm': None,
        'right_wrist': None,
        'left_arm': None,
        'left_wrist': None,
        'leg': None,
        'right_foot': None,
        'left_foot': None 
    }
    
    
    #pm = pix / mm, 画像の1ピクセルあたりの実距離mmを概算したもの
    '''
    顔
    →鼻の位置が検出されている
    '''
    if nose[r] > 0.1:
        
        #両耳の位置が検出されている場合
        if right_ear[r] > 0.1 and left_ear[r] > 0.1:
            #検出結果から両耳間の距離を計算
            ear_dist = m.dist((right_ear[x], right_ear[y]), (left_ear[x], left_ear[y]))
            #pm: 耳珠間幅(A3: 145.7)を用いて算出
            pm = ear_dist / 145.7
            
            '''
            左右端
            '''
            #検出した耳の位置から耳の幅(A20: 31.0)だけずらした位置を左右端とする．
            face_x0 = max(0, right_ear[x] - pm*(31.0 + beta))
            face_x1 = min(w, left_ear[x]  + pm*(31.0 + beta))
            
            '''
            上下端
            '''
            #頭頂・耳珠距離(A33: 135.4)を用いて耳の位置から頭頂部の位置を概算する
            face_y0 = max(0, min(right_ear[y], left_ear[y])- pm*(135.4 + beta))
            #全頭高(A36: 234.0)を用いて頭頂部から顎の位置を概算する
            face_y1 = min(h, face_y0 + pm*(234 + 2*beta))
        
        #右耳の位置は検出されているが左耳の位置は検出されていない場合        
        elif right_ear[r] > 0.1 and left_ear[r] < 0.1:
            #耳~鼻の距離を計算
            ear_nose = m.dist((right_ear[x], right_ear[y]), (nose[x], nose[y]))
            #pm: 鼻~頭部後端(A21: 199.5)と耳~頭部後端(A27: 87.4)から算出
            pm = ear_nose / (199.5 - 87.4)
            
            '''
            左端
            '''
            face_x0 = max(0, right_ear[x] - pm*(31.0 + beta))
            
            '''
            右端
            '''
            #左目が検出されている場合
            if left_eye[r] > 0.1:
                #耳珠間幅(A3: 145.7)と瞳孔間幅(A9: 60.7)を用いて左目の位置から左耳の位置を概算
                face_x1 = min(w, left_eye[x] + pm*((145.7 - 60.7)/2)+ pm*beta)
                
            #左目が検出されていない場合
            elif left_eye[r] < 0.1:
                face_x1 = min(w, nose[x] + pm*beta)
                
            '''
            上端
            '''
            #右耳の位置から概算した頭頂部の位置
            face_y0 = max(0, right_ear[y] - pm*(135.4 + beta))
            
            '''
            下端
            '''
            #頭頂部から概算した顎の位置
            face_y1 = min(h, face_y0 + pm*(234 + 2*beta))
            
        #左耳の位置は検出されているが右耳の位置は検出されていない場合
        elif left_ear[r] > 0.1 and right_ear[r] < 0.1:
            #耳~鼻の距離を計算
            ear_nose = m.dist((left_ear[x], left_ear[y]), (nose[x], nose[y]))
            #pm: 鼻~頭部後端(A21: 199.5)と耳~頭部後端(A27: 87.4)から算出
            pm = ear_nose / (199.5 - 87.4)
            
            '''
            左端
            '''
            #耳の幅．耳介間幅(A4: 185.7)と耳珠間幅(A3: 145.7)を用いて計算
            ear_w = pm*(185.7 - 145.7) / 2
            face_x0 = left_ear[x] - (ear_w + pm*beta)
            
            '''
            右端
            '''
            #右目が検出されている場合
            if right_ear[r] > 0.1:
                #耳珠間幅(A3: 145.7)と瞳孔間幅(A9: 60.7)を用いて左目の位置から左耳の位置を概算
                face_x1 = right_eye[x] + pm*((145.7 - 60.7)/2) + pm*beta
                
            #右目が検出されていない場合
            elif right_ear[r] <= 0.1:
                face_x1 = nose[x] + pm*beta
                
            '''
            上端
            '''
            #左耳の位置から概算した頭頂部の位置
            face_y0 = max(0, left_ear[y]-pm*(135.4+beta))
            
            '''
            下端
            '''
            #頭頂部から概算した顎の位置
            face_y1 = min(h, face_y0 + pm*(234 + 2*beta))
                        
        
        #顔の位置
        coordinates['face'] = [face_x0, face_x1, face_y0, face_y1]
        
    
    '''
    後頭部
    →両耳の位置が検出されていて，鼻の位置が検出されていない
    '''
    if right_ear[r] > 0.1 and left_ear[r] > 0.1 and nose[r] < 0.1:
        #検出結果から両耳間の距離を計算
        ear_dist = m.dist((right_ear[x], right_ear[y]), (left_ear[x], left_ear[y]))
        #pm: 耳珠間幅(A3: 145.7)を用いて算出
        pm = ear_dist / 145.7
        
        '''
        左右端
        '''
        #検出した耳の位置から耳幅の分ずらした所
        backhead_x0 = max(0, left_ear[x] - pm*(31.0 + beta))
        backhead_x1 = min(w, right_ear[x] + pm*(31.0 + beta))
        
        '''
        上端
        '''
        #頭頂・耳珠距離(A33: 135.4)を用いて耳の位置から頭頂部の位置を概算する
        backhead_y0 = max(0, min(right_ear[y], left_ear[y])- pm*(135.4 + beta))
        
        '''
        下端
        ''' 
        #心臓の位置が検出されている場合
        if heart[r] > 0.1:
            backhead_y1 = heart[y]
            
        #心臓の位置が検出されていない場合
        elif heart[y] < 0.1:
            #身長(B1: 1654.7)と腋窩高(B9: 1220.6)を用いて心臓の位置を概算
            backhead_y1 = backhead_y0 + pm*(1654.7 - 1220.6 + beta)
            
        
        #後頭部の位置
        coordinates['back_head'] = [backhead_x0, backhead_x1, backhead_y0, backhead_y1]
        
        
    '''
    胸腹部
    →両肩の位置が検出されていて，右肩が左肩よりも左側にある
    '''
    if right_shoulder[r] > 0.1 and left_shoulder[r] > 0.1 and right_shoulder[x] < left_shoulder[x]:
        #検出結果から肩間の距離を計算
        shoulder_dist = m.dist((right_shoulder[x], right_shoulder[y]), \
                             (left_shoulder[x], left_shoulder[y]))
        #pm: 肩峰幅(D7: 378.8)を用いて算出
        pm = shoulder_dist / 378.8
        
        #肩幅(D2: 432.8)と肩峰幅を用いて肩の中心~外側の距離を計算
        shoulder_w = (432.8 - 378.8) / 2
        
        #両肘の位置が検出されている場合
        if right_elbow[r] > 0.1 and left_elbow[r] > 0.1:
            '''
            左右端
            '''
            #肩と肘のうちより端にある部位の位置から肩の外側までの距離分ずらす
            chest_x0 = max(0, min(right_shoulder[x], right_elbow[x])-(shoulder_w+beta)*pm)
            chest_x1 = min(w, max(left_shoulder[x], left_elbow[x])+(shoulder_w+beta)*pm)
        
        #右肘の位置は検出されているが左肘の位置は推定されていない場合
        elif right_elbow[r] > 0.1 and left_elbow[r] < 0.1:
            '''
            左右端
            '''
            #右端は上と同じ
            chest_x0 = max(0, min(right_shoulder[x], right_elbow[x])-(shoulder_w+beta)*pm)
            chest_x1 = min(w, left_shoulder[x]+(shoulder_w+beta)*pm)
            
        #左肘の位置は検出されているが右肘の位置は推定されていない場合
        elif left_elbow[r] > 0.1 and right_elbow[r] < 0.1:
            '''
            左右端
            '''
            #左端は上と同じ
            chest_x0 = max(0, right_shoulder[x]-shoulder_w*pm-beta)
            chest_x1 = min(w, left_shoulder[x]+shoulder_w*pm+beta)
        
        
        '''
        上下端
        '''
        #肩峰高(B19: 1331.0)と頸椎高(B8:1405.1)を用いて心臓~首元の距離を概算．
        chest_y0 = heart[y] - pm*(1405.1 - 1331.0 + beta) 
        chest_y1 = center_waist[y] + pm*beta

        
        coordinates['chest'] = [chest_x0, chest_x1, chest_y0, chest_y1]
        
        
        
    '''
    背部
    →両肩の位置が検出されていて左肩が右肩より左側にある
    '''
    if (right_shoulder[r] > 0.1 and left_shoulder[r] > 0.1) and (right_shoulder[x] > left_shoulder[x]):
        #検出結果から肩間の距離を算出
        shoulder_dist = m.dist((right_shoulder[x], right_shoulder[y]), (left_shoulder[x], left_shoulder[y]))
        #pm: 肩峰幅(D7: 378.8)を用いて算出
        pm = shoulder_dist / 378.8
        
        #肩幅(D2: 432.8)と肩峰幅を用いて肩の中心~外側の距離を計算
        shoulder_w = (432.8 - 378.8) / 2
        
        #両肘の位置が検出されている場合
        if right_elbow[r] > 0.1 and left_elbow[r] > 0.1:
            '''
            左右端
            '''
            #肩と肘のうちより端にある部位の位置から肩の外側までの距離分ずらす
            back_x0 = max(0, min(left_shoulder[x], left_elbow[x])-(shoulder_w+beta)*pm)
            back_x1 = min(w, max(right_shoulder[x], right_elbow[x])+(shoulder_w+beta)*pm)
        
        #右肘の位置は検出されているが左肘の位置は推定されていない場合
        elif right_elbow[r] > 0.1 and left_elbow[r] < 0.1:
            '''
            左右端
            '''
            #右端は上と同じ
            back_x0 = max(0, min(left_shoulder[x], left_elbow[x])-(shoulder_w+beta)*pm)
            back_x1 = min(w, right_shoulder[x]+(shoulder_w+beta)*pm)
            
        #左肘の位置は検出されているが右肘の位置は推定されていない場合
        elif left_elbow[r] > 0.1 and right_elbow[r] < 0.1:
            '''
            左右端
            '''
            #左端は上と同じ
            back_x0 = max(0, left_shoulder[x]-shoulder_w*pm-beta)
            back_x1 = min(w, right_shoulder[x]+shoulder_w*pm+beta)
        
        
        '''
        上下端
        '''
        #肩峰高(B19: 1331.0)と頸椎高(B8:1405.1)を用いて心臓~首元の距離を概算．
        back_y0 = heart[y] - pm*(1405.1 - 1331.0 + beta) 
        back_y1 = center_waist[y] + pm*beta

        
        coordinates['back'] = [back_x0, back_x1, back_y0, back_y1]
        
        
        
    '''
    右腕
    →右肩と右肘の位置が検出されている
    '''
    if right_shoulder[r] > 0.1 and right_elbow[r] > 0.1:
        #検出結果から上腕の長さ計算
        arm_l = m.dist((right_shoulder[x], right_shoulder[y]), (right_elbow[x], right_elbow[y]))
        #上肢長(C7: 301.2)を用いて計算
        pm = arm_l / 301.2
        
        #右手首の位置が検出されている場合
        if right_wrist[r] > 0.1:
            '''
            左右端
            '''
            #肩，肘，手首のうち最も端にあるものから腕付根前後径(E7: 112.4)だけずらす
            rightarm_x0 = max(0, min(right_shoulder[x], right_elbow[x], right_wrist[x])-pm*(112.4+beta))
            rightarm_x1 = min(w, max(right_shoulder[x], right_elbow[x], right_wrist[x])+pm*(112.4+beta))
            
            '''
            上下端
            '''
            #左右端と同じ感じ
            rightarm_y0 = max(0, min(right_shoulder[y], right_elbow[y], right_wrist[y])-pm*(112.4+beta))
            rightarm_y1 = min(h, max(right_shoulder[y], right_elbow[y], right_wrist[y])+pm*(112.4+beta))
            
        #右手首の位置が検出されていない場合
        elif right_wrist[r] < 0.1:
            '''
            左右端
            '''
            #肩，肘のうち端にあるものから腕付根前後径(E7: 112.4)だけずらす
            rightarm_x0 = max(0, min(right_shoulder[x], right_elbow[x])-pm*(112.4+beta))
            rightarm_x1 = min(w, max(right_shoulder[x], right_elbow[x])+pm*(112.4+beta))
            
            '''
            上下端
            '''
            #左右端と同じ感じ
            rightarm_y0 = max(0, min(right_shoulder[y], right_elbow[y])-pm*(112.4+beta))
            rightarm_y1 = min(h, max(right_shoulder[y], right_elbow[y])+pm*(112.4+beta))
            
            
        coordinates['right_arm'] = [rightarm_x0, rightarm_x1, rightarm_y0, rightarm_y1]
        
       
        
    '''
    右手首
    →右手首の位置が検出されている
    '''
    if right_wrist[r] > 0.1:
        #右肘の位置が検出されている場合
        if right_elbow[r] > 0.1:
            #検出結果から前腕の長さを計算
            arm_l = m.dist((right_wrist[x], right_wrist[y]), (right_elbow[x], right_elbow[y]))
            #前腕長(C8: 240.5)を用いて計算
            pm = arm_l / 240.5
            
        #右肘の位置は検出されていないが，左肘と左手首の位置が検出されている場合
        elif right_elbow[r] < 0.1 and (left_elbow[r] > 0.1 and left_wrist[r]):
            #検出結果から前腕の長さを計算
            arm_l = m.dist((left_wrist[x], left_wrist[y]), (left_elbow[x], left_elbow[y]))
            #前腕長(C8: 240.5)を用いて計算
            pm = arm_l / 240.5
            
        '''
        左右端・上下端
        '''
        #手首の位置から第三指手長(L1: 182.6)の長さだけずらす
        rightwrist_x0 = max(0, right_wrist[x] - pm*(182.6 + beta))
        rightwrist_x1 = min(w, right_wrist[x] + pm*(182.6 + beta))
        rightwrist_y0 = max(0, right_wrist[y] - pm*(182.6 + beta))
        rightwrist_y1 = min(h, right_wrist[y] + pm*(182.6 + beta))
        
            
        coordinates['right_wrist'] = [rightwrist_x0, rightwrist_x1, rightwrist_y0, rightwrist_y1]
        
        
    
    '''
    左腕
    →左肩と左肘の位置が検出されている
    '''
    if left_shoulder[r] > 0.1 and left_elbow[r] > 0.1:
        #検出結果から上腕の長さ計算
        arm_l = m.dist((left_shoulder[x], left_shoulder[y]), (left_elbow[x], left_elbow[y]))
        #上肢長(C7: 301.2)を用いて計算
        pm = arm_l / 301.2
        
        #右手首の位置が検出されている場合
        if left_wrist[r] > 0.1:
            '''
            左右端
            '''
            #肩，肘，手首のうち最も端にあるものから腕付根前後径(E7: 112.4)だけずらす
            leftarm_x0 = max(0, min(left_shoulder[x], left_elbow[x], left_wrist[x])-pm*(112.4+beta))
            leftarm_x1 = min(w, max(left_shoulder[x], left_elbow[x], left_wrist[x])+pm*(112.4+beta))
            
            '''
            上下端
            '''
            #左右端と同じ感じ
            leftarm_y0 = max(0, min(left_shoulder[y], left_elbow[y], left_wrist[y])-pm*(112.4+beta))
            leftarm_y1 = min(h, max(left_shoulder[y], left_elbow[y], left_wrist[y])+pm*(112.4+beta))
            
        #右手首の位置が検出されていない場合
        elif left_wrist[r] < 0.1:
            '''
            左右端
            '''
            #肩，肘のうち端にあるものから腕付根前後径(E7: 112.4)だけずらす
            leftarm_x0 = max(0, min(left_shoulder[x], left_elbow[x])-pm*(112.4+beta))
            leftarm_x1 = min(w, max(left_shoulder[x], left_elbow[x])+pm*(112.4+beta))
            
            '''
            上下端
            '''
            #左右端と同じ感じ
            leftarm_y0 = max(0, min(left_shoulder[y], left_elbow[y])-pm*(112.4+beta))
            leftarm_y1 = min(h, max(left_shoulder[y], left_elbow[y])+pm*(112.4+beta))
            
            
        coordinates['left_arm'] = [leftarm_x0, leftarm_x1, leftarm_y0, leftarm_y1]
        
       
        
    '''
    左手首
    →左手首の位置が検出されている
    '''
    if left_wrist[r] > 0.1:
        #左肘の位置が検出されている場合
        if left_elbow[r] > 0.1:
            #検出結果から前腕の長さを計算
            arm_l = m.dist((left_wrist[x], left_wrist[y]), (left_elbow[x], left_elbow[y]))
            #前腕長(C8: 240.5)を用いて計算
            pm = arm_l / 240.5
            
        #左肘の位置は検出されていないが，右肘と右手首の位置が検出されている場合
        elif left_elbow[r] < 0.1 and (right_elbow[r] > 0.1 and right_wrist[r]):
            #検出結果から前腕の長さを計算
            arm_l = m.dist((right_wrist[x], right_wrist[y]), (right_elbow[x], right_elbow[y]))
            #前腕長(C8: 240.5)を用いて計算
            pm = arm_l / 240.5
            
        '''
        左右端・上下端
        '''
        #手首の位置から第三指手長(L1: 182.6)の長さだけずらす
        leftwrist_x0 = max(0, left_wrist[x] - pm*(182.6 + beta))
        leftwrist_x1 = min(w, left_wrist[x] + pm*(182.6 + beta))
        leftwrist_y0 = max(0, left_wrist[y] - pm*(182.6 + beta))
        leftwrist_y1 = min(h, left_wrist[y] + pm*(182.6 + beta))
        
            
        coordinates['left_wrist'] = [leftwrist_x0, leftwrist_x1, leftwrist_y0, leftwrist_y1]
        
        
        
    '''
    脚部
    →左右腰と膝の位置が検出されている
    '''
    if right_waist[r] > 0.1 and left_waist[r] > 0.1 and right_knee[r] > 0.1 and left_knee[r] > 0.1:
        #検出結果から左右それぞれの大腿の長さを計算し，平均を取る
        right_thigh_l = m.dist((right_waist[x], right_waist[y]), (right_knee[x], right_knee[y]))
        left_thigh_l = m.dist((left_waist[x], left_waist[y]), (right_knee[x], right_knee[y]))
        thigh_l = np.mean([right_thigh_l, left_thigh_l])
        
        #大腿長(C13: 403.4)を用いて計算
        pm = thigh_l / 403.4
        #大腿の幅と厚さの平均
        thigh = np.mean([164.3, 174.3])
        
        '''
        上端
        '''
        #中央腰の位置が検出されている場合
        if center_waist[r] > 0.1:
            leg_y0 = max(0, min(center_waist[y], right_waist[y], left_waist[y])-pm*beta)
        
        #中央腰の位置が検出されていない場合
        elif center_waist[r] < 0.1:
            leg_y0 = max(0, min(right_waist[y], left_waist[y])-pm*beta)
            
        
        '''
        左右端・下端
        '''
        #両足首の位置が検出されている場合
        if right_ankle[r] > 0.1 and left_ankle[r] > 0.1:
            #腰，膝，足首のうち最も端にあるものの位置から大腿幅(D15: 164.3)と大腿厚(E8: 174.3)の平均だけずらす
            leg_x0 = max(0, min(right_waist[x], right_knee[x], right_ankle[x], left_waist[x], left_knee[x], left_ankle[x])-pm*(thigh+beta))
            leg_x1 = min(w, max(right_waist[x], right_knee[x], right_ankle[x], left_waist[x], left_knee[x], left_ankle[x])+pm*(thigh+beta))
            
            #腰，膝，足首のうち最も下にあるものの位置から内果高(M2: 79.2)分だけずらす
            leg_y1 = min(h, max(right_waist[y], right_knee[y], right_ankle[y], left_waist[y], left_knee[y], left_ankle[y])+pm*(79.2+beta))
            
        #右足首の位置は検出されているが，左足首の位置は検出されていない場合
        elif right_ankle[r] > 0.1 and left_ankle[r] < 0.1:
            #腰，膝，足首のうち最も端にあるものの位置から大腿幅(D15: 164.3)と大腿厚(E8: 174.3)の平均だけずらす
            leg_x0 = max(0, min(right_waist[x], right_knee[x], right_ankle[x], left_waist[x], left_knee[x])-pm*(thigh+beta))
            leg_x1 = min(w, max(right_waist[x], right_knee[x], right_ankle[x], left_waist[x], left_knee[x])+pm*(thigh+beta))
            
            #腰，膝，足首のうち最も下にあるものの位置から内果高(M2: 79.2)分だけずらす
            leg_y1 = min(h, max(right_waist[y], right_knee[y], right_ankle[y], left_waist[y], left_knee[y])+pm*(79.2+beta))
            
        #左足首の位置は検出されているが，右足首の位置は検出されていない場合
        elif left_ankle[r] > 0.1 and right_ankle[r] < 0.1:
            #腰，膝，足首のうち最も端にあるものの位置から大腿幅(D15: 164.3)と大腿厚(E8: 174.3)の平均だけずらす
            leg_x0 = max(0, min(right_waist[x], right_knee[x], left_waist[x], left_knee[x], left_ankle[x])-pm*(thigh+beta))
            leg_x1 = min(w, max(right_waist[x], right_knee[x], left_waist[x], left_knee[x], left_ankle[x])+pm*(thigh+beta))
            
            #腰，膝，足首のうち最も下にあるものの位置から内果高(M2: 79.2)分だけずらす
            leg_y1 = max(h, max(right_waist[y], right_knee[y], left_waist[y], left_knee[y], left_ankle[y])+pm*(79.2+beta))
            
        #両足首の位置が検出されていない場合
        elif right_ankle[r] < 0.1 and left_ankle[r] < 0.1:
            #膝の位置と下肢長(C14: 374.2)を用いて足首の位置を概算
            ankle_y = max(right_knee[y], left_knee[y]) + pm*374.2
            
            leg_y1 = min(h, ankle_y+pm*beta)
            
            
        coordinates['leg'] = [leg_x0, leg_x1, leg_y0, leg_y1]
        
        
        
    '''
    右足
    →右足首，右つま先(内外どちらか)，右かかとの位置が検出されている
    '''
    if (right_ankle[r] > 0.1 and right_heel[r] > 0.1) and (right_toe_in[r] > 0.1 or right_toe_out[r] > 0.1):
        #つま先の内側の信頼度が外側よりも高い場合
        if right_toe_in[r] >= right_toe_out[r]:
            #足の長さ
            foot_l = m.dist((right_toe_in[x], right_toe_in[y]), (right_heel[x], right_heel[y]))
            #足長(M15: 243.9)を用いて計算
            pm = foot_l / 243.9
            
        #つま先の内側の信頼度が外側よりも高い場合
        elif right_toe_out[r] > right_toe_in[r]:
            #足の長さ
            foot_l = m.dist((right_toe_out[x], right_toe_out[y]), (right_heel[x], right_heel[y]))
            #足長(M15: 243.9)を用いて計算
            pm = foot_l / 243.9    
            
        #内不踏長(M: 179.0)と外不踏長(M21: 156.3)の平均
        instep_l = np.mean([179.0, 156.3])
        
        #つま先の内側と外側の位置が検出されている場合
        if right_toe_in[r] > 0.1 and right_toe_out[r] > 0.1:
            '''
            左右端
            '''
            #つま先内外，足首，かかとのうち最も端にあるものの位置から不踏長さの平均分ずらす
            rightfoot_x0 = max(0, min(right_ankle[x], right_toe_in[x], right_toe_out[x], right_heel[x])-pm*(instep_l+beta))
            rightfoot_x1 = min(w, max(right_ankle[x], right_toe_in[x], right_toe_out[x], right_heel[x])+pm*(instep_l+beta))
            
            '''
            上下端
            '''
            #つま先内外，足首，かかとのうち最も端にあるものの位置から内くるぶし高さ(M2: 79.2)分ずらす
            rightfoot_y0 = max(0, min(right_ankle[y], right_toe_in[y], right_toe_out[y], right_heel[y])-pm*(79.2+beta))
            rightfoot_y1 = min(h, max(right_ankle[y], right_toe_in[y], right_toe_out[y], right_heel[y])+pm*(79.2+beta))
            
        #つま先の内側の位置は検出されているが，外側は検出されていない場合
        elif right_toe_in[r] > 0.1 and right_toe_out[r] < 0.1:
            '''
            左右端
            '''
            #つま先，足首，かかとのうち最も端にあるものの位置から不踏長さの平均分ずらす
            rightfoot_x0 = max(0, min(right_ankle[x], right_toe_in[x], right_heel[x])-pm*(instep_l+beta))
            rightfoot_x1 = min(w, max(right_ankle[x], right_toe_in[x], right_heel[x])+pm*(instep_l+beta))
            
            '''
            上下端
            '''
            #つま先，足首，かかとのうち最も端にあるものの位置から内くるぶし高さ(M2: 79.2)分ずらす
            rightfoot_y0 = max(0, min(right_ankle[y], right_toe_in[y], right_heel[y])-pm*(79.2+beta))
            rightfoot_y1 = min(h, max(right_ankle[y], right_toe_in[y], right_heel[y])+pm*(79.2+beta))
            
        #つま先の外側の位置は検出されているが，内側の位置は検出されていない場合
        elif right_toe_out[r] > 0.1 and right_toe_in[r] < 0.1:
            '''
            左右端
            '''
            #つま先，足首，かかとのうち最も端にあるものの位置から不踏長さの平均分ずらす
            rightfoot_x0 = max(0, min(right_ankle[x], right_toe_out[x], right_heel[x])-pm*(instep_l+beta))
            rightfoot_x1 = min(w, max(right_ankle[x], right_toe_out[x], right_heel[x])+pm*(instep_l+beta))
            
            '''
            上下端
            '''
            #つま先，足首，かかとのうち最も端にあるものの位置から内くるぶし高さ(M2: 79.2)分ずらす
            rightfoot_y0 = max(0, min(right_ankle[y], right_toe_out[y], right_heel[y])-pm*(79.2+beta))
            rightfoot_y1 = min(h, max(right_ankle[y], right_toe_out[y], right_heel[y])+pm*(79.2+beta))
            
            
        coordinates['right_foot'] = [rightfoot_x0, rightfoot_x1, rightfoot_y0, rightfoot_y1]
        
        
            
    '''
    左足
    →左足首，左つま先(内外どちらか)，左かかとの位置が検出されている
    '''
    if (left_ankle[r] > 0.1 and left_heel[r] > 0.1) and (left_toe_in[r] > 0.1 or left_toe_out[r] > 0.1):
        #つま先の内側の信頼度が外側よりも高い場合
        if left_toe_in[r] >= left_toe_out[r]:
            #足の長さ
            foot_l = m.dist((left_toe_in[x], left_toe_in[y]), (left_heel[x], left_heel[y]))
            #足長(M15: 243.9)を用いて計算
            pm = foot_l / 243.9
            
        #つま先の内側の信頼度が外側よりも高い場合
        elif left_toe_out[r] > left_toe_in[r]:
            #足の長さ
            foot_l = m.dist((left_toe_out[x], left_toe_out[y]), (left_heel[x], left_heel[y]))
            #足長(M15: 243.9)を用いて計算
            pm = foot_l / 243.9    
            
        #内不踏長(M: 179.0)と外不踏長(M21: 156.3)の平均
        instep_l = np.mean([179.0, 156.3])
        
        #つま先の内側と外側の位置が検出されている場合
        if left_toe_in[r] > 0.1 and left_toe_out[r] > 0.1:
            '''
            左右端
            '''
            #つま先内外，足首，かかとのうち最も端にあるものの位置から不踏長さの平均分ずらす
            leftfoot_x0 = max(0, min(left_ankle[x], left_toe_in[x], left_toe_out[x], left_heel[x])-pm*(instep_l+beta))
            leftfoot_x1 = min(w, max(left_ankle[x], left_toe_in[x], left_toe_out[x], left_heel[x])+pm*(instep_l+beta))
            
            '''
            上下端
            '''
            #つま先内外，足首，かかとのうち最も端にあるものの位置から内くるぶし高さ(M2: 79.2)分ずらす
            leftfoot_y0 = max(0, min(left_ankle[y], left_toe_in[y], left_toe_out[y], left_heel[y])-pm*(79.2+beta))
            leftfoot_y1 = min(h, max(left_ankle[y], left_toe_in[y], left_toe_out[y], left_heel[y])+pm*(79.2+beta))
            
        #つま先の内側の位置は検出されているが，外側は検出されていない場合
        elif left_toe_in[r] > 0.1 and left_toe_out[r] < 0.1:
            '''
            左右端
            '''
            #つま先，足首，かかとのうち最も端にあるものの位置から不踏長さの平均分ずらす
            leftfoot_x0 = max(0, min(left_ankle[x], left_toe_in[x], left_heel[x])-pm*(instep_l+beta))
            leftfoot_x1 = min(w, max(left_ankle[x], left_toe_in[x], left_heel[x])+pm*(instep_l+beta))
            
            '''
            上下端
            '''
            #つま先，足首，かかとのうち最も端にあるものの位置から内くるぶし高さ(M2: 79.2)分ずらす
            leftfoot_y0 = max(0, min(left_ankle[y], left_toe_in[y], left_heel[y])-pm*(79.2+beta))
            leftfoot_y1 = min(h, max(left_ankle[y], left_toe_in[y], left_heel[y])+pm*(79.2+beta))
            
        #つま先の外側の位置は検出されているが，内側の位置は検出されていない場合
        elif left_toe_out[r] > 0.1 and left_toe_in[r] < 0.1:
            '''
            左右端
            '''
            #つま先，足首，かかとのうち最も端にあるものの位置から不踏長さの平均分ずらす
            leftfoot_x0 = max(0, min(left_ankle[x], left_toe_out[x], left_heel[x])-pm*(instep_l+beta))
            leftfoot_x1 = min(w, max(left_ankle[x], left_toe_out[x], left_heel[x])+pm*(instep_l+beta))
            
            '''
            上下端
            '''
            #つま先，足首，かかとのうち最も端にあるものの位置から内くるぶし高さ(M2: 79.2)分ずらす
            leftfoot_y0 = max(0, min(left_ankle[y], left_toe_out[y], left_heel[y])-pm*(79.2+beta))
            leftfoot_y1 = min(h, max(left_ankle[y], left_toe_out[y], left_heel[y])+pm*(79.2+beta))
        
        
        coordinates['left_foot'] = [leftfoot_x0, leftfoot_x1, leftfoot_y0, leftfoot_y1]
        
        
    return coordinates       
    

'''
身体部位画像を作成する関数
'''
def make_part_image(image: np.ndarray, keypoints: np.ndarray or None):
    '''
    Parameters
    ----------
    image : np.ndarray
        人物画像 / カメラ画像
    keypoints : np.ndarray or None
        キーポイントの検出結果

    Returns
    -------
    part_images: dict
        身体部位の画像

    '''
    
    #キーポイントが検出されてなければ検出
    if type(keypoints) == None:
        keypoints = detect_keypoints(image)
        
    #画像の高さと幅
    h, w, _ = image.shape
    
    #部位ごとに切り取る位置を決める
    coordinates = decide_coordinates(keypoints[0], w, h)
    
    #切り取った画像を入れていく辞書
    part_images = {}
    #身体部位ごとのループ
    for part in coordinates.keys():

        try:
            #身体部位の領域が小さかったらその部位の画像は作成しない．
            if coordinates[part][3] - coordinates[part][2] < 30 and coordinates[part][1] - coordinates[part][0] < 30:
                part_images[part] = None
                
            else:
                part_images[part] = image[round(coordinates[part][2]): round(coordinates[part][3]), \
                                           round(coordinates[part][0]): round(coordinates[part][1])]
        
        except TypeError:
            continue
            
    
    return part_images

        
    
    
    



