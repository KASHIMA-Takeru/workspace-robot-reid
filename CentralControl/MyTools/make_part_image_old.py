# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:34:34 2023

@author: ab19109
OpenPoseで推定した座標を基に全身画像から身体部位ごとに画像を切り取る関数
"""
import numpy as np
import cv2
'''
切り取り位置を決める関数
'''
def decide_coordinates(keys, w, h):
    # =============================================================================
    # keys(list): OpenPoseで推定したキーポイントの座標が入ったリスト
    # w, h(int): 画像の幅と高さ
    # =============================================================================
    
    #左上のxy座標(x0, y0), 右下のxy座標(x1, y1)とする．
    #部位をまとめた辞書を作成し，座標のリスト[x0, x1, y0, y1]を値として更新する．
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
    
    #pm = pix / mm
    '''
    顔
    →鼻の位置が推定されている
    '''
    if keys[0][2] > 0.1:
        #両耳の位置が推定されている場合
        if keys[17][2] > 0.1 and keys[18][2] > 0.1:
            #pm: 両耳間の距離(A3: 145.7)から算出
            pm = (keys[18][0] - keys[17][0]) / 145.7
            
            #左右端
            #両耳の外側を左右端に設定する
            face_x0 = keys[17][0]
            face_x1 = keys[18][0]
            
            #顔の長さを概算．頭頂部~耳と頭頂部~顎を用いる
            #上のy座標 右耳，左耳のうち上にある方~頭頂部まで(A33: 135.4)
            face_y0 = max(0, min(keys[17][1], keys[18][1]) - pm * 135.4)
            #下のy座標　頭頂部~顎 (A36: 234.0)
            face_y1 = min(h, face_y0 + pm * 234.0)
            
        #片耳のみ推定されている場合
        #右耳は推定されているが左耳は推定されていない場合
        elif keys[17][2] > 0.1 and keys[18][2] < 0.1:
            #pm: 鼻~頭部後端(A21: 199.5)と耳~頭部後端(A27: 87.4)から算出 
            pm = (keys[0][0] - keys[17][0]) / (199.5 - 87.4)
            
            face_x0 = keys[17][0]
            
            #左目が推定されている場合
            if keys[16][2] > 0.1:
                #右端のx座標: 左目のx座標から左耳の位置を概算
                #耳~目: 両耳間の距離(A3: 145.7)と目間の距離(A9: 60.7)から算出
                face_x1 = keys[16][0] + pm * (145.7 - 60.7) * 0.5 
                
            #左目が推定されていない場合
            else:
                face_x1 = keys[0][0]
                
            #上のy座標　右耳~頭頂部(A33: 135.4)
            face_y0 = max(0, keys[17][1] - pm * 135.4)
            #下のy座標　頭頂部~顎 (A36: 234.0)
            face_y1 = min(h, face_y0 + pm * 234.0)
            
        #左耳は推定されているが，右耳は推定されていない場合
        elif keys[18][2] > 0.1 and keys[17][2] < 0.1:
            #鼻~頭部後端(A21: 199.5)と耳~頭部後端(A27: 87.4)から算出 
            pm = (keys[18][0] - keys[0][0]) / (199.5 - 87.4)
            
            face_x1 = keys[18][0]
            
            #右目が推定されている場合
            if keys[15][2] > 0.1:
                #右端のx座標: 右目のx座標から右耳の位置を概算
                face_x0 = keys[15][0] - pm * (145.7 - 60.7) * 0.5
                
            #右目が推定されていない場合
            else:
                face_x0 = keys[0][0]
            
            #上のy座標　右耳~頭頂部(A33: 135.4)
            face_y0 = keys[18][1] - pm * 135.4
            #下のy座標　頭頂部~顎 (A36: 234.0)
            face_y1 = face_y0 + pm * 234.0
            
        #両耳が推定されていない場合
        elif keys[17][2] < 0.1 and keys[18][2] < 0.1:
            face_x0 = None
            face_x1 = None
            face_y0 = None
            face_y1 = None
            
        #顔の位置
        coordinates['face'] = [face_x0, face_x1, face_y0, face_y1]
    
    
    '''
    後頭部
    →両耳の位置が推定されていて鼻の位置が推定されていない
    '''
    if ((keys[17][2] > 0.1 and keys[18][2] > 0.1) and (keys[0][2] < 0.1)):
        #pm: 両耳間の距離(A3: 145.7)から算出
        pm = (keys[17][0] - keys[18][0]) / 145.7
        
        #顔の長さを概算．頭頂部~耳と頭頂部~顎を用いる
        #上のy座標 右耳，左耳のうち上にある方~頭頂部まで(A33: 135.4)
        backhead_y0 = min(keys[17][1], keys[18][1]) - pm * 135.4
        
        #下のy座標について．心臓の位置が推定されている場合
        if keys[1][2] > 0.1:
            backhead_y1 = keys[1][1]
            
        else:
            #下のy座標　頭頂部~心臓辺り 床~頭頂部( = 身長)(B1: 1654.7)と,床~腋窩(B9: 1220.6)から算出
            backhead_y1 = backhead_y0 + pm * (1654.7 - 1220.6)
            
        #後頭部の位置
        coordinates['back_head'] = [keys[18][0], keys[17][0], backhead_y0, backhead_y1]
        
    
    '''
    胸腹部
    →両肩の位置が推定されており，右肩が左肩より左側にある
    '''
    if ((keys[2][2] > 0.1 and keys[5][2]) and (keys[2][0] < keys[5][0])):
        #pm: 両肩の幅(D2: 432.8)から算出
        pm = (keys[5][0] - keys[2][0]) / 432.8
        
        #両肘の位置が推定されている場合
        if keys[3][2] > 0.1 and keys[6][2] > 0.1:
            #左右端: 肩と肘のうち端にある方の位置
            chest_x0 = min(keys[2][0], keys[3][0])
            chest_x1 = max(keys[5][0], keys[6][0])
            
        #両肘の位置が推定されていない場合
        else:
            chest_x0 = keys[2][0]
            chest_x1 = keys[5][0]
        
        #上下端(上端： 首の根本辺り)
        chest_y0 = keys[1][1] - pm * (1405.1 - 1220.6)
        chest_y1 = keys[8][1]
        
        coordinates['chest'] = [chest_x0, chest_x1, chest_y0, chest_y1]

    
    '''
    背部
    →両肩の位置が推定されており，左肩が右肩より左側にある
    '''
    if ((keys[2][2] > 0.1 and keys[5][2]) and (keys[5][0] < keys[2][0])):
        #pm: 両肩の幅(D2: 432.8)から算出
        pm = (keys[2][0] - keys[5][0]) / 432.8
        
        #左右端
        back_x0 = keys[5][0]
        back_x1 = keys[2][0]
        
        #上下端(上端： 首の根本辺り)
        back_y0 = max(0, keys[1][1] - pm * (1405.1 - 1220.6))
        back_y1 = keys[8][1]
        
        coordinates['back'] = [back_x0, back_x1, back_y0, back_y1]

    
    '''
    右腕
    →右肩の位置が推定されている
    '''
    if keys[2][2] > 0.1:
        #肘と手首の位置がともに推定されている場合
        if keys[3][2] > 0.1 and keys[4][2] > 0.1:
            #pm: 肩~肘の長さから算出(C10: 326.4)
            pm = np.sqrt((keys[2][0] - keys[3][0]) ** 2 + (keys[2][1] - keys[3][1]) ** 2) / 326.4
            
            #左右端: 肩，肘，手首のうち一番端(左右)にある部位の位置から前腕最大幅(D14: 88.5)を足し引きする
            rightarm_x0 = max(0, min(keys[2][0], keys[3][0], keys[4][0]) - pm * 88.5)
            rightarm_x1 = min(w, max(keys[2][0], keys[3][0], keys[4][0]) + pm * 88.5)
            
            #上下端: 肩，肘，手首のうち一番端(上下)にある部位の位置から前腕最大幅(D14: 88.5)を足し引きする
            rightarm_y0 = max(0, min(keys[2][1], keys[3][1], keys[4][1]) - pm * 88.5)
            rightarm_y1 = min(h, max(keys[2][1], keys[3][1], keys[4][1]) + pm * 88.5)
            
        #肘の位置は推定されているが手首の位置は推定されていない場合
        if keys[3][2] > 0.1 and keys[4][2] < 0.1:
            #pm: 上と同じ
            pm = np.sqrt((keys[2][0] - keys[3][0]) ** 2 + (keys[2][1] - keys[3][1]) ** 2) / 326.4

            
            #左右端: 肩と肘のうち端にある方の位置から前腕最大幅を足し引きする．
            rightarm_x0 = max(0, min(keys[2][0], keys[3][0]) - pm * 88.5)
            rightarm_x1 = min(w, max(keys[2][0], keys[3][0]) + pm * 88.5)
            
            #上下端: 肩と肘のうち端(上下)にある方の位置から前腕最大幅を足し引きする．
            rightarm_y0 = max(0, min(keys[2][1], keys[3][1]) - pm * 88.5)
            rightarm_y1 = min(h, max(keys[2][1], keys[3][1]) + pm * 88.5)
            
        #肘の位置は推定されていないが，手首の位置は推定されている場合
        if keys[3][2] < 0.1 and keys[4][2] > 0.1:
            #pm: 上腕長さ + 前腕長さ(C7: 301.2 + C8: 240.5)から概算
            pm = np.sqrt((keys[2][0] - keys[4][0]) ** 2 + (keys[2][1] - keys[4][1]) ** 2) / (301.2 + 240.5)

            
            #左右端
            rightarm_x0 = max(0, min(keys[2][0], keys[4][0]) - pm * 88.5)
            rightarm_x1 = min(w, min(keys[2][0], keys[4][0]) + pm * 88.5)
            
            #上下端
            rightarm_y0 = max(0, min(keys[2][1], keys[4][1]) - pm * 88.5)
            rightarm_y1 = min(h, max(keys[2][1], keys[4][1]) + pm * 88.5)
            
        #肘と手首の位置がともに推定されていない場合
        if keys[3][2] < 0.1 and keys[4][2] < 0.1:
            rightarm_x0 = None
            rightarm_x1 = None
            rightarm_y0 = None
            rightarm_y1 = None
            
        
        coordinates['right_arm'] = [rightarm_x0, rightarm_x1, rightarm_y0, rightarm_y1]
        
    
    '''
    右手首
    →右手首の位置が推定されている
    '''
    if keys[4][2] > 0.1:
        #肘の位置が推定されている場合
        if keys[3][2] > 0.1:
            #pm: 前腕長さ(C8: 240.5)から算出
            pm = np.sqrt((keys[4][0] - keys[3][0]) ** 2 + (keys[4][1] - keys[3][1]) ** 2) / 240.5
        
        #右肘の位置は推定されていないが，左肘と左手首の位置が推定されている場合
        elif keys[3][2] < 0.1 and keys[6][2] > 0.1 and keys[7][2] > 0.1:
            pm = np.sqrt((keys[6][0] - keys[7][0]) ** 2 + (keys[6][1] - keys[7][1]) ** 2) / 240.5
        
        #左右端，上下端: 手首の位置から手の長さ(C9: 183.5)を足し引きする
        rightwrist_x0 = max(0, keys[4][0] - pm * 183.5)
        rightwrist_x1 = min(w, keys[4][0] + pm * 183.5)
        
        rightwrist_y0 = max(0, keys[4][1] - pm * 183.5)
        rightwrist_y1 = min(h, keys[4][1] + pm * 183.5)
            
        
        coordinates['right_wrist'] = [rightwrist_x0, rightwrist_x1, rightwrist_y0, rightwrist_y1]
        
    '''
    左腕
    →左肩の位置が推定されている
    '''
    if keys[5][2] > 0.1:
        #肘と手首の位置がともに推定されている場合
        if keys[6][2] > 0.1 and keys[7][2] > 0.1:
            #pm: 肩~肘の長さから算出(C10: 326.4)
            pm = np.sqrt((keys[5][0] - keys[6][0]) ** 2 + (keys[5][1] - keys[6][1]) ** 2) / 326.4
            
            #左右端: 肩，肘，手首のうち一番端(左右)にある部位の位置から前腕最大幅(D14: 88.5)を足し引きする
            leftarm_x0 = max(0, min(keys[5][0], keys[6][0], keys[7][0]) - pm * 88.5)
            leftarm_x1 = min(w, max(keys[5][0], keys[6][0], keys[7][0]) + pm * 88.5)
            
            #上下端: 肩，肘，手首のうち一番端(上下)にある部位の位置から前腕最大幅(D14: 88.5)を足し引きする
            leftarm_y0 = max(0, min(keys[5][1], keys[6][1], keys[7][1]) - pm * 88.5)
            leftarm_y1 = min(h, max(keys[5][1], keys[6][1], keys[7][1]) + pm * 88.5)
            
        #肘の位置は推定されているが手首の位置は推定されていない場合
        if keys[6][2] > 0.1 and keys[7][2] < 0.1:
            #pm: 上と同じ
            pm = np.sqrt((keys[5][0] - keys[6][0]) ** 2 + (keys[5][1] - keys[6][1]) ** 2) / 326.4

            
            #左右端: 肩と肘のうち端にある方の位置から前腕最大幅を足し引きする．
            leftarm_x0 = max(0, min(keys[5][0], keys[6][0]) - pm * 88.5)
            leftarm_x1 = min(w, max(keys[5][0], keys[6][0]) + pm * 88.5)
            
            #上下端: 肩と肘のうち端(上下)にある方の位置から前腕最大幅を足し引きする．
            leftarm_y0 = max(0, min(keys[5][1], keys[6][1]) - pm * 88.5)
            leftarm_y1 = min(h, max(keys[5][1], keys[6][1]) + pm * 88.5)
            
        #肘の位置は推定されていないが，手首の位置は推定されている場合
        if keys[6][2] < 0.1 and keys[7][2] > 0.1:
            #pm: 上腕長さ + 前腕長さ(C7: 301.2 + C8: 240.5)から概算
            pm = np.sqrt((keys[5][0] - keys[7][0]) ** 2 + (keys[5][1] - keys[7][1]) ** 2) / (301.2 + 240.5)

            
            #左右端
            leftarm_x0 = max(0, min(keys[5][0], keys[7][0]) - pm * 88.5)
            leftarm_x1 = min(w, min(keys[5][0], keys[7][0]) + pm * 88.5)
            
            #上下端
            leftarm_y0 = max(0, min(keys[5][1], keys[7][1]) - pm * 88.5)
            leftarm_y1 = min(h, max(keys[5][1], keys[7][1]) + pm * 88.5)
            
        #肘と手首の位置がともに推定されていない場合
        if keys[6][2] < 0.1 and keys[7][2] < 0.1:
            leftarm_x0 = None
            leftarm_x1 = None
            leftarm_y0 = None
            leftarm_y1 = None
            
        
        coordinates['left_arm'] = [leftarm_x0, leftarm_x1, leftarm_y0, leftarm_y1]
        
    
    '''
    左手首
    →左手首の位置が推定されている
    '''
    if keys[7][2] > 0.1:
        #肘の位置が推定されている場合
        if keys[6][2] > 0.1:
            #pm: 前腕長さ(C8: 240.5)から算出
            pm = np.sqrt((keys[7][0] - keys[6][0]) ** 2 + (keys[7][1] - keys[6][1]) ** 2) / 240.5
            
        #左肘の位置は推定されていないが，右肘と右手首の位置が推定されている場合
        elif keys[6][2] < 0.1 and keys[3][2] > 0.1 and keys[4][2] > 0.1:
            pm = np.sqrt((keys[3][0] - keys[4][0]) ** 2 + (keys[3][1] - keys[4][1]) ** 2) / 240.5
            
        #左右端，上下端: 手首の位置から手の長さ(C9: 183.5)を足し引きする
        leftwrist_x0 = max(0, keys[7][0] - pm * 183.5)
        leftwrist_x1 = min(w, keys[7][0] + pm * 183.5)
        
        leftwrist_y0 = max(0, keys[7][1] - pm * 183.5)
        leftwrist_y1 = min(h, keys[7][1] + pm * 183.5)
            
        
        coordinates['left_wrist'] = [leftwrist_x0, leftwrist_x1, leftwrist_y0, leftwrist_y1]
    
    
    '''
    脚部
    →左右の腰と膝の位置が推定されている
    '''
    if keys[9][2] > 0.1 and keys[10][2] > 0.1 and  keys[12][2] > 0.1 and keys[13][2] > 0.1:
        #pm: 大腿長さ(C13: 403.0)から算出．左右の平均をとる
        pm = np.mean(np.sqrt((keys[9][0]-keys[10][0])**2 + (keys[9][1]-keys[10][1])**2) +\
                     np.sqrt((keys[12][0]-keys[13][0])**2 + (keys[12][1]-keys[13][1])**2)) / 403.0 
            
        #両足首の位置が推定されている場合
        if keys[11][2] > 0.1 and keys[14][2] > 0.1:
            #左右端: 腰，膝，足首のうち端にある部位の位置から大腿幅(D15: 164.3を足し引きする)
            leg_x0 = max(0, min(keys[9][0], keys[10][0], keys[11][0], keys[12][0], keys[13][0], keys[14][0]) - pm * 164.3 * 0.5)
            leg_x1 = min(w, max(keys[9][0], keys[10][0], keys[11][0], keys[12][0], keys[13][0], keys[14][0]) + pm * 164.3 * 0.5)
            
            leg_y0 = min(keys[9][1], keys[10][1], keys[11][1], keys[12][1], keys[13][1], keys[14][1])
            leg_y1 = max(keys[11][1], keys[14][1])
            
        #片足のみ推定されている場合
        #右足首のみ推定されている場合
        elif keys[11][2] > 0.1 and keys[14][2] < 0.1:
            leg_x0 = max(0, min(keys[9][0], keys[10][0], keys[11][0], keys[12][0], keys[13][0]) - pm * 164.3 * 0.5)
            leg_x1 = min(w, max(keys[9][0], keys[10][0], keys[11][0], keys[12][0], keys[13][0]) + pm * 164.3 * 0.5)
            
            leg_y0 = min(keys[9][1], keys[10][1], keys[11][1], keys[12][1], keys[13][1])
            leg_y1 = keys[11][1]
        
        #左足首のみ推定されている場合
        elif keys[11][2] < 0.1 and keys[14][2] > 0.1:
            leg_x0 = max(0, min(keys[9][0], keys[10][0], keys[12][0], keys[13][0], keys[14][0]) - pm * 164.3 * 0.5)
            leg_x1 = min(w, max(keys[9][0], keys[10][0], keys[12][0], keys[13][0], keys[14][0]) + pm * 164.3 * 0.5)
            
            leg_y0 = min(keys[9][1], keys[10][1], keys[12][1], keys[13][1], keys[14][1])
            leg_y1 = keys[14][2]
            
        #両足首の位置が推定されていない場合
        elif keys[11][2] < 0.1 and keys[14][2] < 0.1:
            leg_x0 = max(0, min(keys[9][0], keys[10][0], keys[12][0], keys[13][0]) - pm * 164.3 * 0.5)
            leg_x1 = min(w, max(keys[9][0], keys[10][0], keys[12][0], keys[13][0]) + pm * 164.3 * 0.5)
            
            leg_y0 = min(keys[9][1], keys[10][1], keys[12][1], keys[13][1])
            leg_y1 = max(keys[10][1], keys[13][1])
            
        coordinates['leg'] = [leg_x0, leg_x1, leg_y0, leg_y1]
        
    
    '''
    右足
    →右足首，右つま先(内外どちらか)，右かかとの位置が推定されている
    '''
    if (keys[11][2] > 0.1 and keys[24][2] > 0.1) and (keys[22][2] > 0.1 or keys[23][2] > 0.1):
        #つま先の内側の信頼度が外側の信頼度より高い場合
        if keys[22][2] >= keys[23][2]:
            #pm: 足長さ(M19: 244.0)から算出．
            pm = np.sqrt((keys[22][0] - keys[24][0]) ** 2 + (keys[22][1] - keys[24][1]) ** 2) / 244.0
            
        #つま先の外側の信頼度が内側より高い場合
        elif keys[23][2] > keys[22][2]:
            #pm: 足長さ(M19: 244.0)から算出．
            pm = np.sqrt((keys[23][0] - keys[24][0]) ** 2 + (keys[23][1] - keys[24][1]) ** 2) / 244.0
            
        #つま先の内側と外側の位置が推定されている場合
        if keys[22][2] > 0.1 and keys[23][2] > 0.1:
            #左右端: 足首，つま先，かかとのうち端にある部位の位置から不踏長さ(M20: 179.0, M21: 156.3の平均)を足し引きする
            rightfoot_x0 = max(0, min(keys[11][0], keys[22][0], keys[23][0], keys[24][0]) - pm * (179.0 + 156.3) * 0.5)
            rightfoot_x1 = min(w, max(keys[11][0], keys[22][0], keys[23][0], keys[24][0]) + pm * (179.0 + 156.3) * 0.5)
            
            #上下端: 足首，つま先，かかとのうち下にある部位の位置から内くるぶし高さ(M2: 79.2)を足し引きする
            rightfoot_y0 = max(0, min(keys[11][1], keys[22][1], keys[23][1], keys[24][1]) - pm * 79.2)
            rightfoot_y1 = min(h, max(keys[11][1], keys[22][1], keys[23][1], keys[24][1]) + pm * 79.2)
            
        #内側のみ推定されている場合
        elif keys[22][2] > 0.1 and keys[23][2] < 0.1:
            #左右端: 足首，つま先，かかとのうち端にある部位の位置から不踏長さ(M20: 179.0, M21: 156.3の平均)を足し引きする
            rightfoot_x0 = max(0, min(keys[11][0], keys[22][0], keys[24][0]) - pm * (179.0 + 156.3) * 0.5)
            rightfoot_x1 = min(w, max(keys[11][0], keys[22][0], keys[24][0]) + pm * (179.0 + 156.3) * 0.5)
            
            #上下端: 足首，つま先，かかとのうち下にある部位の位置から内くるぶし高さ(M2: 79.2)を足し引きする
            rightfoot_y0 = max(0, min(keys[11][1], keys[22][1], keys[24][1]) - pm * 79.2)
            rightfoot_y1 = min(h, max(keys[11][1], keys[22][1], keys[24][1]) + pm * 79.2)            
        
        #外側のみ推定されている場合
        elif keys[23][2] > 0.1 and keys[22][2] < 0.1:
            #左右端: 足首，つま先，かかとのうち端にある部位の位置から不踏長さ(M20: 179.0, M21: 156.3の平均)を足し引きする
            rightfoot_x0 = max(0, min(keys[11][0], keys[23][0], keys[24][0]) - pm * (179.0 + 156.3) * 0.5)
            rightfoot_x1 = min(w, max(keys[11][0], keys[23][0], keys[24][0]) + pm * (179.0 + 156.3) * 0.5)
            
            #上下端: 足首，つま先，かかとのうち下にある部位の位置から内くるぶし高さ(M2: 79.2)を足し引きする
            rightfoot_y0 = max(0, min(keys[11][1], keys[23][1], keys[24][1]) - pm * 79.2)
            rightfoot_y1 = min(h, max(keys[11][1], keys[23][1], keys[24][1]) + pm * 79.2)            
        
        
        coordinates['right_foot'] = [rightfoot_x0, rightfoot_x1, rightfoot_y0, rightfoot_y1]
        
    '''
    左足
    →左足首，左つま先(内外どちらか)，左かかとの位置が推定されている
    '''
    if (keys[14][2] > 0.1 and keys[21][2] > 0.1) and (keys[19][2] > 0.1 or keys[20][2] > 0.1):
        #つま先の内側の信頼度が外側の信頼度より高い場合
        if keys[19][2] >= keys[20][2]:
            #pm: 足長さ(M19: 244.0)から算出．
            pm = np.sqrt((keys[19][0] - keys[21][0]) ** 2 + (keys[19][1] - keys[21][1]) ** 2) / 244.0
            
        #つま先の外側の信頼度が内側より高い場合
        elif keys[20][2] > keys[19][2]:
            #pm: 足長さ(M19: 244.0)から算出．
            pm = np.sqrt((keys[20][0] - keys[21][0]) ** 2 + (keys[20][1] - keys[21][1]) ** 2) / 244.0
                        
        #つま先の内側と外側の位置が推定されている場合
        if keys[19][2] > 0.1 and keys[20][2] > 0.1:
            
            #左右端: 足首，つま先，かかとのうち端にある部位の位置から不踏長さ(M20: 179.0, M21: 156.3の平均)を足し引きする
            leftfoot_x0 = max(0, min(keys[14][0], keys[19][0], keys[20][0], keys[21][0]) - pm * (179.0 + 156.3) * 0.5)
            leftfoot_x1 = min(w, max(keys[14][0], keys[19][0], keys[20][0], keys[21][0]) + pm * (179.0 + 156.3) * 0.5)
            
            #上下端: 足首，つま先，かかとのうち下にある部位の位置から外くるぶし高さ(M4: 67.8)を足し引きする
            leftfoot_y0 = max(0, min(keys[14][1], keys[19][1], keys[20][1], keys[21][1]) - pm * 67.8)
            leftfoot_y1 = min(h, max(keys[14][1], keys[19][1], keys[20][1], keys[21][1]) + pm * 67.8)
        
        #内側のみ推定されている場合
        elif keys[19][2] > 0.1 and keys[20][2] < 0.1:
            #左右端: 足首，つま先，かかとのうち端にある部位の位置から不踏長さ(M20: 179.0, M21: 156.3の平均)を足し引きする
            leftfoot_x0 = max(0, min(keys[14][0], keys[19][0], keys[21][0]) - pm * (179.0 + 156.3) * 0.5)
            leftfoot_x1 = min(w, max(keys[14][0], keys[19][0], keys[21][0]) + pm * (179.0 + 156.3) * 0.5)
            
            #上下端: 足首，つま先，かかとのうち下にある部位の位置から外くるぶし高さ(M4: 67.8)を足し引きする
            leftfoot_y0 = max(0, min(keys[14][1], keys[19][1], keys[21][1]) - pm * 67.8)
            leftfoot_y1 = min(h, max(keys[14][1], keys[19][1], keys[21][1]) + pm * 67.8)
    
        #外側のみ推定されている場合
        elif keys[20][2] > 0.1 and keys[19][2] < 0.1:
            #左右端: 足首，つま先，かかとのうち端にある部位の位置から不踏長さ(M20: 179.0, M21: 156.3の平均)を足し引きする
            leftfoot_x0 = max(0, min(keys[14][0], keys[20][0], keys[21][0]) - pm * (179.0 + 156.3) * 0.5)
            leftfoot_x1 = min(w, max(keys[14][0], keys[20][0], keys[21][0]) + pm * (179.0 + 156.3) * 0.5)
            
            #上下端: 足首，つま先，かかとのうち下にある部位の位置から外くるぶし高さ(M4: 67.8)を足し引きする
            leftfoot_y0 = max(0, min(keys[14][1], keys[20][1], keys[21][1]) - pm * 67.8)
            leftfoot_y1 = min(h, max(keys[14][1], keys[20][1], keys[21][1]) + pm * 67.8)            
            
            
            
        coordinates['left_foot'] = [leftfoot_x0, leftfoot_x1, leftfoot_y0, leftfoot_y1]

    return coordinates
        

def crop_by_part(image, keys):
    # =============================================================================
    # image (str / ndarray): 処理を行う画像．OpenCVで読み込む
    # keys (list): OpenPoseで推定したキーポイントの座標が入ったリスト    
    # =============================================================================
    
    #imageがstr(パス)なら読み込む
    if isinstance(image, str):
        img = cv2.imread(image)
        
    #imageが既に読み込まれている
    elif isinstance(image, np.ndarray):
        img = image
        
    h, w = img.shape[:2]

    #print("Keys > ", keys)
    coordinates = decide_coordinates(keys, w, h)
    #print("coordinates > ", coordinates)
        
    #切り取った画像を入れる辞書
    cropped_images = {}
    #身体部位ごとに切り取って辞書に追加
    for part in coordinates.keys():
        try:
            #身体部位の面積が小さかったらその部位の画像は作成しない
            if (coordinates[part][3] - coordinates[part][2]) * (coordinates[part][1] - coordinates[part][0]) < 2000:
                cropped_images[part] = None
            
            else:
                cropped_images[part] = img[round(coordinates[part][2]): round(coordinates[part][3]), \
                                                          round(coordinates[part][0]): round(coordinates[part][1])]
                            
        #切り取る座標がNoneの場合
        except TypeError:
            cropped_images[part] = None
                
    return cropped_images
            