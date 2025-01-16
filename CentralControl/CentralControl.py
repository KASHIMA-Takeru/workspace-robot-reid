#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python -*-

# <rtc-template block="description">
"""
 @file CentralControl.py
 @brief operate Re-ID and robot control
 @date $Date$


"""
# </rtc-template>
import sys
import time



sys.path.append(".")
sys.path.append(r"C:\Users\ab19109\workspce_robot_reid")

# Import RTM module
import RTC
import OpenRTM_aist

import cv2
import numpy as np
import math as m
import random as rd

import os
import os.path as osp

import json
import base64
import csv

import datetime
import time

from MyTools import openpose_processor as opp

from reid_base import ReIDBase

# Import Service implementation class
# <rtc-template block="service_impl">

# </rtc-template>

# Import Service stub modules
# <rtc-template block="consumer_import">
# </rtc-template>


# This module's spesification
# <rtc-template block="module_spec">
centralcontrol_spec = ["implementation_id", "CentralControl", 
         "type_name",         "CentralControl", 
         "description",       "operate Re-ID and robot control", 
         "version",           "1.0.0", 
         "vendor",            "Kashima", 
         "category",          "Controller", 
         "activity_type",     "STATIC", 
         "max_instance",      "1", 
         "language",          "Python", 
         "lang_type",         "SCRIPT",
         ""]
# </rtc-template>

# <rtc-template block="component_description">
##
# @class CentralControl
# @brief operate Re-ID and robot control
# 
# 
# </rtc-template>
class CentralControl(OpenRTM_aist.DataFlowComponentBase):
    
    ##
    # @brief constructor
    # @param manager Maneger Object
    # 
    def __init__(self, manager):
        OpenRTM_aist.DataFlowComponentBase.__init__(self, manager)
        
        #カラー画像
        self._d_image_data = RTC.CameraImage(RTC.Time(0, 0), 0, 0, 0, "", 0.0, "")
        self._image_dataIn = OpenRTM_aist.InPort("image_data", self._d_image_data)
        
        #深度画像
        self._d_depth_data = RTC.CameraImage(RTC.Time(0, 0), 0, 0, 0, "", 0.0, "")
        self._depth_dataIn = OpenRTM_aist.InPort("depth_data", self._d_depth_data)

        #ロボットの動作指令
        self._d_motion_instruct = RTC.TimedVelocity2D(RTC.Time(0,0), RTC.Velocity2D(0.0, 0.0, 0.0))
        self._motion_instructionOut = OpenRTM_aist.OutPort("motion_instruction", self._d_motion_instruct)

        '''
        Re-IDに用いるものたち
        '''
        #Re-IDに使う値などをまとめた辞書
        #weight: ベクトル間の距離にかける重み
        #path: 学習済みモデルのパス
        #model_name: CNNの名前(種類)．モデル読み込み時に使用
        #size: CNNに入力する画像サイズ．(height, width)
        self.pivod_dict = {
            'wholebody': {
                'path': r'E:\md23036\Model\best_models\wholebody\model.pth.tar-22',
                'model_name': 'osnet_highres1',
                'size': (512, 256)
                },
            'face': {
                'weight': 2.0,
                'path': r'E:\md23036\Model\best_models\part\model_face_3.pth.tar-4',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'back_head': {
                'weight': 0.25,
                'path': r'E:\md23036\Model\best_models\part\model_backhead_1.pth.tar-8',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'chest': {
                'weight': 0.25,
                'path': r'E:\md23036\Model\best_models\part\model_chest_addblock_dellarge_2.pth.tar-24',
                'model_name': 'osnet_part_addblock_dellarge',
                'size': (256, 128)
                },
            'back': {
                'weight': 0.25,
                'path': r'E:\md23036\Model\best_models\part\model_back_5.pth.tar-22',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'right_arm': {
                'weight': 1.0,
                'path': r'E:\md23036\Model\best_models\part\model_right_arm_2.pth.tar-2',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'right_wrist': {
                'weight': 1.5,
                'path': r'E:\md23036\Model\best_models\part\model_right_wrist_delsmall_5.pth.tar-24',
                'model_name': 'osnet_part_delsmall',
                'size': (256, 128)
                },
            'left_arm': {
                'weight': 1.0,
                'path': r'E:\md23036\Model\best_models\part\model_left_arm_4.pth.tar-25',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'left_wrist': {
                'weight': 1.5,
                'path': r'E:\md23036\Model\best_models\part\model_left_wrist_3.pth.tar-5',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'leg': {
                'weight': 0.1,
                'path': r'E:\md23036\Model\best_models\part\model_leg_4.pth.tar-6',
                'model_name': 'osnet',
                'size': (256, 128)
                },
            'right_foot': {
                'weight': 2.0,
                'path': r'E:\md23036\Model\best_models\part\model_right_foot_resize_5.pth.tar-19',
                'model_name': 'osnet',
                'size': (64, 128)
                },
            'left_foot': {
                'weight': 2.0,
                'path': r'E:\md23036\Model\best_models\part\model_left_foot_resize_2.pth.tar-23',
                'model_name': 'osnet',
                'size': (64, 128)
                }
            }
     
        #追尾対象のID
        self.target_id = '022'
        self.id_list = ['015', '016', '022', '024']
        #self.id_list = ['001', '002']
        #保存フォルダ101
        #D:\master_research\Robot\gallery_storage\1121 1213
        self.save_folder = r'D:\master_research\Robot\experiment1220'
        #検索データの保存先
        self.gallery_folder = r'D:\master_research\Robot\gallery_storage\1213'
        #NNのパス
        self.nn_path = r'C:\Users\ab19109\workspce_robot_reid\CentralControl\nn_model.pth'

        #Re-IDを実行する頻度(frame / 回)
        self.reid_freq = 20
        
        #身体部位画像でのRe-IDを行うか
        self.use_part = False
        #同一人物かの判断にNNを使うか
        self.use_nn = False
        #同一人物か異なる人物かを判断する閾値
        self.thrs = 300
        #探索する最大人数
        self.maxk = 50

        '''
        ロボットの制御に使用するものたち
        '''
        #ロボットの基本移動速度[m/s]
        self.normal_speed = 0.3
        #ロボットの基本回転速度
        self.va = 0.1

        #ロボットの低速移動時の速度
        self.slow_speeds = [0.2, 0.1]
        #ロボットが低速移動を始める目安となる人とロボットの距離[m]
        self.slow_dist = [2.0, 1.0]

        #ロボットと人の最短距離(この距離以下になったらロボットを停止させる)[m]
        self.min_dist = 0.7

        #カメラ画像の左・中央・右の境界
        self.l_border = 240
        self.r_border = 400

        '''
        使用するカメラに関する値
        '''
        #画像の高さと幅
        self.height = 480
        self.width = 640
        
        #色の定義
        self.BLUE = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.WHITE = (255, 255, 255)

        #追尾対象が直前まで左右どちら側にいたか
        self.last_time = 'center'


        self.target_x = int(self.width / 2)
        self.target_y = int(self.height / 2)
        # initialize of configuration-data.
        # <rtc-template block="init_conf_param">
        
        # </rtc-template>


         
    ##
    #
    # The initialize action (on CREATED->ALIVE transition)
    # 
    # @return RTC::ReturnCode_t
    # 
    #
    def onInitialize(self):

        # Bind variables and configuration variable
        
        # Set InPort buffers
        self.addInPort("image_data", self._image_dataIn)
        self.addInPort("depth_data", self._depth_dataIn)
        
        # Set OutPort buffers
        self.addOutPort("motion_instruction", self._motion_instructionOut)
        

        # Set service provider to Ports
        
        # Set service consumers to Ports
        
        # Set CORBA Service Ports

        #Re-IDクラスのオブジェクト生成
        self.reid = ReIDBase(
            pivod_dict = self.pivod_dict, 
            save_dir = self.save_folder,
            use_part = self.use_part,
            use_nn = self.use_nn,
            thrs = self.thrs,
            maxk = self.maxk
            )

        #Re-ID準備
        self.reid.prepare(self.gallery_folder, self.nn_path, self.id_list)

        
        print("Ready for Re-ID")
        
        return RTC.RTC_OK
    
    ###
    ## 
    ## The finalize action (on ALIVE->END transition)
    ## 
    ## @return RTC::ReturnCode_t
    #
    ## 
    #def onFinalize(self):
    #

    #    return RTC.RTC_OK
    
    ###
    ##
    ## The startup action when ExecutionContext startup
    ## 
    ## @param ec_id target ExecutionContext Id
    ##
    ## @return RTC::ReturnCode_t
    ##
    ##
    #def onStartup(self, ec_id):
    #
    #    return RTC.RTC_OK
    
    ###
    ##
    ## The shutdown action when ExecutionContext stop
    ##
    ## @param ec_id target ExecutionContext Id
    ##
    ## @return RTC::ReturnCode_t
    ##
    ##
    #def onShutdown(self, ec_id):
    #
    #    return RTC.RTC_OK
    
    ##
    #
    # The activated action (Active state entry action)
    #
    # @param ec_id target ExecutionContext Id
    # 
    # @return RTC::ReturnCode_t
    #
    #
    def onActivated(self, ec_id):
        #結果記録用に実行時刻を取得(月日_時分秒の形に変換)
        exe_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
        #保存先作成
        self.save_path = osp.join(self.save_folder, exe_time)
        os.makedirs(self.save_path, exist_ok=True)

        #条件のメモ用のtxtファイル
        self.f = open(osp.join(self.save_path, 'memo.txt'), 'a')
        print("Execution Date: ", exe_time, file=self.f)
        print("==============", file=self.f)
        print("Basic Informatoin", file=self.f)
        print("--------------", file=self.f)
        #print(self.pivod_dict, file=self.f)
        print("Threshold > ", self.thrs, file=self.f)
        print("Use body part: ", self.use_part, file=self.f)
        print("Use NN: ", self.use_nn, file=self.f)
        print("\n- Robot speed", file=self.f)
        print("Normal speed: ", self.normal_speed, file=self.f)
        print("1st slow speed: {} m/s (less than {} m)".format(self.slow_speeds[0], self.slow_dist[0]), file=self.f)
        print("2nd slow speed: {} m/s (less than {} m)".format(self.slow_speeds[1], self.slow_dist[1]), file=self.f)
        print("Minimum distance: {} m\n".format(self.min_dist), file=self.f)
        print("Re-ID frequency: {} frame / time".format(self.reid_freq), file=self.f)

        #記録用のcsvファイル
        logf = open(osp.join(self.save_path, 'log.csv'), 'a', newline="")
        self.csv_writer = csv.writer(logf)
        #ヘッダー書込み
        #self.header_dict = 
        self.csv_writer.writerow(['Frame', 'Decode(RGB)', 'Keypoints detect', 'Make person img', 'Re-ID', 'Draw circle', 'Instruct va', \
            'Draw rectangle', 'Decode(depth)', 'Measure dist', 'Instruct vx', 'Imshow', '1cycle'])

        self.n_frame = 0
        self.start_time = time.perf_counter()
        #print("Start: ", self.start_time)

        #画像表示用ウィンドウ
        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)

        #動画ライター準備
        #カラー画像と深度画像を横並びにするため幅は2倍
        width = self.width*2
        height = self.height
        #FPSは事前に測定したものに近い値に設定
        fps = 2
        fmt = cv2.VideoWriter_fourcc(*'mp4v')

        self.writer = cv2.VideoWriter(osp.join(self.save_path, 'result.mp4'), fmt, fps, (width, height))

        #Re-IDを行った回数
        self.n_reid = 0
        #追尾対象を認識したか
        self.target_flag = False
        #ループの最初かどうか(onExecuteで使用)
        self.first_flag = True

        print("Activate")


        return RTC.RTC_OK
    
    ##
    #
    # The deactivated action (Active state exit action)
    #
    # @param ec_id target ExecutionContext Id
    #
    # @return RTC::ReturnCode_t
    #
    #
    def onDeactivated(self, ec_id):
        cv2.destroyAllWindows()
        self.writer.release()
        self.f.close()
        self.csv_writer.release()

        print("Deactivate")
    
        return RTC.RTC_OK
    
    ##
    #
    # The execution action that is invoked periodically
    #
    # @param ec_id target ExecutionContext Id
    #
    # @return RTC::ReturnCode_t
    #
    #
    def onExecute(self, ec_id):
        try:
            time_start_exe = time.perf_counter()
            #カラー画像・深度画像を受け取ったか
            color_img_flag = False
            depth_img_flag = False
            #人物画像が作成されたか
            made_person_flag = False

            #csv書込み用のリスト
            row = []

            '''
            カラー画像に関する処理
            '''
            if self._image_dataIn.isNew():
                time_start_decode = time.perf_counter()
            

                #カラー画像のデコード
                self._d_image_data = self._image_dataIn.read()
                received_data = self._d_image_data.pixels
                data_json = json.loads(received_data)
                image_dec = base64.b64decode(data_json)
                data_np = np.frombuffer(image_dec, dtype='uint8')
                color_image = cv2.imdecode(data_np, 1)

                color_img_flag = True
            
                time_decode = time.perf_counter() - time_start_decode
                #row.append(time_decode)
        
                #OpenPoseで人物検出
                time_start_key = time.perf_counter()
                keypoints, keyimage = opp.detect_keypoints(color_image)
                time_key = time.perf_counter() - time_start_key
                #row.append(time_key)
            
                #keypoints: 検出された人物のキーポイントの座標が入った配列．
                #key_image: キーポイント間を線で結んだ画像
            
                #人物が検出された場合
                if type(keypoints) == np.ndarray:

                    time_start_mpi = time.perf_counter()
                    bbox_list, made_person_flag = opp.make_person_image(image=color_image, keypoints=keypoints)
                    #bbox_list: 人物領域の四隅の座標が入ったリスト

                    #人物画像作成
                    people_list = []
                    for (top, bottom, left, right) in bbox_list:
                        person = color_image[top: bottom, left: right]
                        people_list.append(person)

                    time_mpi = time.perf_counter() - time_start_mpi
                    #row.append(time_mpi)

                #画像の左上に対象人物のIDを書いておく
                cv2.putText(keyimage, 'Target: {}'.format(self.target_id), (5, 20), cv2.FONT_HERSHEY_PLAIN, 2, self.BLUE, thickness=2)


                #人物画像が作成された場合
                if made_person_flag:
                    #指定したフレーム数 or 追尾対象を認識できていないならRe-ID実行
                    if self.n_frame % self.reid_freq == 0 or not self.target_flag:
                        #print("People: ", len(people_list))
                        time_start_reid = time.perf_counter()
                        #Re-ID実行
                        target_index, pid_list, self.target_flag = self.reid.run_reid(people_list, color_image, self.target_id, keypoints)
                        self.n_reid += 1
                        #target_index: OpenPoseで検出した順番が0埋めの文字列として入っている．追尾対象がいないと判断された場合は'Not_exist'が入っている．
                        
                        time_reid = time.perf_counter() - time_start_reid
                        #row.append(time_reid)

                        #追尾対象が見つかった場合
                        if self.target_flag:
                            target_index = int(target_index)
                            time_start_circle = time.perf_counter()
                        
                            #追尾の基準点として対象の心臓部を丸で囲う
                            target_point = keypoints[target_index][1]
                            #追尾対象の心臓部のx, y座標
                            self.target_x = int(target_point[0])
                            self.target_y = int(target_point[1])
                            #print("target x > ", self.target_x)
                            #print("target y > ", self.target_y)

                            #cv2.circle(keyimage, (self.target_x, self.target_y), 25, self.BLUE, thickness=3)
                            time_circle = time.perf_counter() - time_start_circle
                            #row.append(time_circle)
                            '''
                            #検出された人物全員を矩形で囲う
                            for i, (target_box, pid) in enumerate(zip(bbox_list, pid_list)):
                                #人物を囲う色と太さ．追尾対象は青3で他は緑2
                                if i == target_index:
                                    color = self.BLUE
                                    thickness = 3

                                else:
                                    color = self.GREEN
                                    thickness = 2

                                cv2.rectangle(keyimage, (target_box[2], target_box[0]), (target_box[3], target_box[1]), color=color, thickness=thickness)
                                cv2.putText(keyimage, 'No.{}'.format(i), (target_box[2], target_box[0]), cv2.FONT_HERSHEY_PLAIN, 3, color=color, thickness=thickness)
                            '''
                            '''
                            #ロボットへの指令
                            #基準点がカメラ画角の中央より左側にある場合
                            time_start_instruct = time.perf_counter()
                            if target_x < 220:
                                if self.check:
                                    print("direction #1")
                                self._d_motion_instruct.data.va = self.va
                                self.last_time = 'left'

                            #基準点が画角中心より右側にある場合
                            elif target_x > 420:
                                if self.check:
                                    print("direction #2")
                                self._d_motion_instruct.data.va = -1 * self.va
                                self.last_time = 'right'

                            else:
                                if self.check:
                                    print("direction #3")
                                self._d_motion_instruct.data.va = 0
                                self.last_time = 'center'
                            time_instruct = time.perf_counter() - time_start_instruct
                            #row.append(time_instruct)
                            '''

                        #追尾対象が見つからなかった場合
                        else:
                            self._d_motion_instruct.data.vx = 0.0

                            #対象が直前までいた方向に回転する
                            if self.last_time == 'left':
                                self._d_motion_instruct.data.va = 2*self.va

                            elif self.last_time == 'right':
                                self._d_motion_instruct.data.va = -2*self.va
                
                    #Re-IDを行わない場合
                    else:
                        #現在のフレームで検出された人たちの心臓部の位置
                        cur_pos_list = keypoints[:, 1, :2]
                        #print("current position list: ", cur_pos_list)
                        #直前のフレームの追尾対象の位置と現在のフレームで検出された人の位置の距離を計算
                        #print("previous target position: ", self.target_x, self.target_y)
                        diff = np.array([self.target_x, self.target_y]) - cur_pos_list
                        distances = np.linalg.norm(diff, axis=1)
                        #距離が最小の要素の位置
                        target_index = np.argmin(distances)

                        #現在のフレームでの追尾対象の位置
                        cur_target_pos = keypoints[target_index][1]
                        self.target_x = int(cur_target_pos[0])
                        self.target_y = int(cur_target_pos[1])

                    #検出された人物全員を矩形で囲い，追尾の基準点を丸で囲う
                    for i, box in enumerate(bbox_list):
                        #囲う色と線の太さを決め，追尾の基準点を丸で囲う
                        if i == target_index:
                            color = self.BLUE
                            thickness = 3

                            cv2.circle(keyimage, (self.target_x, self.target_y), 10, color=color, thickness=thickness)

                        else:
                            color = self.GREEN
                            thickness = 2

                        cv2.rectangle(keyimage, (box[2], box[0]), (box[3], box[1]), color=color, thickness=thickness)

                
                    #ロボットへの動作指令
                    #基準点がカメラ画角の中央より左側にある場合
                    if self.target_x < self.l_border:

                        self._d_motion_instruct.data.va = self.va
                        self.last_time = 'left'

                    #基準点が画角中心より右側にある場合
                    elif self.target_x > self.r_border:

                        self._d_motion_instruct.data.va = -1 * self.va
                        self.last_time = 'right'

                    else:
                        self._d_motion_instruct.data.va = 0
                        self.last_time = 'center'
                
            
                #人物画像が作成されなかった場合
                else:
                    self.target_flag = False
                    cv2.circle(keyimage, (int(self.width/2), int(self.height/2)), 10, self.BLUE, thickness=3)
                    self._d_motion_instruct.data.vx = 0.0
                    #対象が直前までいた方向に回転する
                    if self.last_time == 'left':
                        self._d_motion_instruct.data.va = 2*self.va

                    elif self.last_time == 'right':
                        self._d_motion_instruct.data.va = -2*self.va

                cv2.putText(keyimage, self.last_time, (250, 20), cv2.FONT_HERSHEY_PLAIN, 2, self.BLUE, thickness=2)
                cv2.putText(keyimage, "Re-ID: {}".format(self.n_reid), (5, 63), cv2.FONT_HERSHEY_PLAIN, 2, self.BLUE, thickness=2)
                color_img_flag = True
        
                #画角の左・中央・右の境界に線を引く
                cv2.line(keyimage, (self.l_border, 0), (self.l_border, self.height), color=self.WHITE, thickness=1)
                cv2.line(keyimage, (self.r_border, 0), (self.r_border, self.height), color=self.WHITE, thickness=1)

            else:
                if self.last_time == 'right':
                    self._d_motion_instruct.data.va = -1 * self.va
                    self.last_time = 'right'

                elif self.last_time == 'left':
                    self._d_motion_instruct.data.va = self.va
                    self.last_time = 'left'

                else:
                    self._d_motion_instruct.data.va = 0.0
                    self.last_time = 'center'
        

            '''
            深度画像に関する処理
            '''
            if self._depth_dataIn.isNew():
                time_start_dec_dep = time.perf_counter()
                #深度データのデコード
                self._d_depth_data = self._depth_dataIn.read()
                #深度データのByte列
                received_depth = self._d_depth_data.pixels
            
                dept_json = json.loads(received_depth)
                dept_dec = base64.b64decode(dept_json)
                dept_np = np.frombuffer(dept_dec, dtype='uint8')
                depth = cv2.imdecode(dept_np, flags=cv2.IMREAD_GRAYSCALE)
                depth_scale = 256*depth

                #カラー画像にする
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_scale, alpha=0.03), cv2.COLORMAP_JET)

                time_dec_dep = time.perf_counter() - time_start_dec_dep
                #row.append(time_dec_dep)

                #対象までの距離
                time_start_dist = time.perf_counter()
                try:
                    target_dist = depth_scale[self.target_y][self.target_x] / 1000

                #対象が検出出来なかった場合の処理
                except:
                    target_dist = 0
                    self.target_x = int(self.width / 2)
                    self.target_y = int(self.height / 2)

                time_dist = time.perf_counter() - time_start_dist
                #row.append(time_dist)


                #対象の位置を円で囲う
                cv2.circle(depth_colormap, (self.target_x, self.target_y), 10, (255, 255, 255), thickness=3)
                cv2.putText(depth_colormap, str(target_dist) + ' m', (5, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)

                depth_img_flag = True

                '''
                ロボットへの移動指令
                '''
                time_start_speed = time.perf_counter()
                if self.target_flag:
                    #対象まで充分な距離がある
                    if target_dist > self.slow_dist[0]:
                        self._d_motion_instruct.data.vx = self.normal_speed

                    #対象に少し近い場合
                    elif (target_dist < self.slow_dist[0]) and (target_dist > self.slow_dist[1]):
                        self._d_motion_instruct.data.vx = self.slow_speeds[0]

                    #更に近い場合
                    elif (target_dist < self.slow_dist[1]) and (target_dist > self.min_dist):
                        self._d_motion_instruct.data.vx = self.slow_speeds[1]

                    #近づきすぎた場合
                    elif target_dist < self.min_dist:
                        self._d_motion_instruct.data.vx = 0.0

                else:
                    self._d_motion_instruct.data.vx = 0.0
                    #self._d_motion_instruct.data.va = 0.0

                time_speed = time.perf_counter() - time_start_speed
                #row.append(time_speed)
                    
                self._motion_instructionOut.write()

            '''
            画像出力
            '''
            if color_img_flag and depth_img_flag:
                #FPS計算
                self.n_frame += 1
                #print("Frame: ", self.n_frame)
                lap = time.perf_counter()
                #print("lap: ", lap)
                laptime = lap - self.start_time
                #print("laptime > ", laptime)

                cur_fps = self.n_frame / laptime
                cv2.putText(depth_colormap, "FPS: {:2f}".format(cur_fps), (5, 63), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
                time_start_imshow = time.perf_counter()

                key_image_dim = keyimage.shape
                depth_map_dim = depth_colormap.shape

                #カラー画像と深度画像を横並びにする
                if key_image_dim != depth_map_dim:
                    resized_keyimage = cv2.resize(keyimage, dsize=(depth_map_dim[1], depth_map_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_keyimage, depth_colormap))

                else:
                    images = np.hstack((keyimage, depth_colormap))

        
                cv2.imshow("Image", images)
                cv2.waitKey(1)

                self.writer.write(images)

                time_imshow = time.perf_counter() - time_start_imshow
                #row.append(time_imshow)

            
            
                print("{} frame, {:3f} FPS".format(self.n_frame, cur_fps), file=self.f)
            
            time_exe = time.perf_counter() - time_start_exe
            #row.append(time_exe)
            #row.insert(self.n_frame)
            #self.csv_writer.writerow(row)

            return RTC.RTC_OK
        
        except Exception as e:
            _, _, trace_back = sys.exc_info()
            filename = trace_back.tb_frame.f_code.co_filename
            line_no = trace_back.tb_lineno

            print("File: {], LiNe No. {}, error: ".format(filename, line_no), e)
    ###
    ##
    ## The aborting action when main logic error occurred.
    ##
    ## @param ec_id target ExecutionContext Id
    ##
    ## @return RTC::ReturnCode_t
    ##
    ##
    #def onAborting(self, ec_id):
    #
    #    return RTC.RTC_OK
    
    ###
    ##
    ## The error action in ERROR state
    ##
    ## @param ec_id target ExecutionContext Id
    ##
    ## @return RTC::ReturnCode_t
    ##
    ##
    #def onError(self, ec_id):
    #
    #    return RTC.RTC_OK
    
    ###
    ##
    ## The reset action that is invoked resetting
    ##
    ## @param ec_id target ExecutionContext Id
    ##
    ## @return RTC::ReturnCode_t
    ##
    ##
    #def onReset(self, ec_id):
    #
    #    return RTC.RTC_OK
    
    ###
    ##
    ## The state update action that is invoked after onExecute() action
    ##
    ## @param ec_id target ExecutionContext Id
    ##
    ## @return RTC::ReturnCode_t
    ##

    ##
    #def onStateUpdate(self, ec_id):
    #
    #    return RTC.RTC_OK
    
    ###
    ##
    ## The action that is invoked when execution context's rate is changed
    ##
    ## @param ec_id target ExecutionContext Id
    ##
    ## @return RTC::ReturnCode_t
    ##
    ##
    #def onRateChanged(self, ec_id):
    #
    #    return RTC.RTC_OK
    



def CentralControlInit(manager):
    profile = OpenRTM_aist.Properties(defaults_str=centralcontrol_spec)
    manager.registerFactory(profile,
                            CentralControl,
                            OpenRTM_aist.Delete)

def MyModuleInit(manager):
    CentralControlInit(manager)

    # create instance_name option for createComponent()
    instance_name = [i for i in sys.argv if "--instance_name=" in i]
    if instance_name:
        args = instance_name[0].replace("--", "?")
    else:
        args = ""
  
    # Create a component
    comp = manager.createComponent("CentralControl" + args)

def main():
    # remove --instance_name= option
    argv = [i for i in sys.argv if not "--instance_name=" in i]
    # Initialize manager
    mgr = OpenRTM_aist.Manager.init(sys.argv)
    mgr.setModuleInitProc(MyModuleInit)
    mgr.activateManager()
    mgr.runManager()

if __name__ == "__main__":
    main()
