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
from turtle import color
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

import datetime


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


        #同一人物か異なる人物かを判断する閾値
        self.thrs = 250
        #追尾対象のID
        self.target_id = '001'
        #保存フォルダ
        self.save_folder = r'D:\master_research\Robot\tracking_test'
        #検索データの保存先
        self.gallery_folder = r'D:\master_research\Robot\gallery_storage'


        '''
        ロボットの制御に使用するものたち
        '''
        #ロボットの基本移動速度[m/s]
        self.normal_speed = 0.3
        #ロボットの基本回転速度
        self.va = 0.2

        #ロボットの低速移動時の速度
        self.slow_speeds = [0.2, 0.1]
        #ロボットが低速移動を始める目安となる人とロボットの距離[m]
        self.slow_dist = [2.0, 1.0]

        #ロボットと人の最短距離(この距離以下になったらロボットを停止させる)[m]
        self.min_dist = 0.5

        #ループの最初かどうか(onExecuteで使用)
        self.first_flag = True

        '''
        使用するカメラに関する値
        '''
        #画像の高さと幅
        self.height = 480
        self.width = 640
        


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
            use_part = True,
            thrs = self.thrs,
            maxk = 20
            )

        #Re-ID準備
        self.reid.prepare(self.gallery_folder)

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

        #画像表示用ウィンドウ
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        #動画ライター準備
        width = self.width*2
        height = self.height
        fps = 4
        fmt = cv2.VideoWriter_fourcc(*'mp4v')

        self.writer = cv2.VideoWriter(osp.join(self.save_path, 'result.mp4'), fmt, fps, (width, height))

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
        #カラー画像・深度画像を受け取ったか
        color_img_flag = False
        depth_img_flag = False
        person_flag = False

        '''
        カラー画像に関する処理
        '''
        if self._image_dataIn.isNew():
            self._d_image_data = self._image_dataIn.read()
            #カラー画像のByte列
            received_data = self._d_image_data.pixels
            #Decode
            data_json = json.loads(received_data)
            image_dec = base64.b64decode(data_json)
            data_np = np.frombuffer(image_dec, dtype='uint8')
            color_image = cv2.imdecode(data_np, 1)

            color_img_flag = True
            print("#1")

            #OpenPoseで人物検出
            key_list, keyimage = opp.detect_keypoints(color_image)
            bbox_list = opp.make_person_image(color_image, key_list)
            people_list = []

            for (top, bottom, left, right) in bbox_list:
                person = color_image[top: bottom, left: right]
                people_list.append(person)

            print("#2")
            #people_list (list): 人物画像のリスト
            #key_list (list): 各人物のキーポイントの検出結果をまとめたリスト
            #keyimage (ndarray): キーポイントを線で結んだ画像

            #人物が検出された場合
            if len(people_list) >= 1:
                print("#3")
                #Re-ID実行
                target_index = self.reid.run_reid(people_list, self.target_id, key_list)
                print("#4")
                #対象人物が見つからなかった場合
                if target_index == None:
                    print("Lost target")
                    #ロボットは移動せずその場で回転．回転方向はランダムに決める
                    if rd.random() > 0.5:
                        self._d_motion_instruct.data.va = self.va

                    else:
                        self._d_motion_instruct.data.va = -1 * self.va

                #対象人物が見つかった場合
                else:
                    print("Detect target")
                    person_flag = True
                    #対象人物の位置を特定
                    #target_indexは，people_listの何番目が対象かをstrで表している
                    target_index = int(target_index)
                    print("target index > ", target_index)
                    target_person = people_list[target_index]

                    #追尾対象の中心位置( = 心臓の位置)
                    target_x = key_list[target_index][1][0]
                    target_y = key_list[target_index][1][0]
                    #対象人物の中心を円で囲う
                    cv2.circle(keyimage, (target_x, target_y), 10, (255, 0, 0), thickness=2)
                    print("#4.1")
                    #目標点( = 追尾対象の中心位置)がカメラ画角より左側にある場合
                    if target_x < 220:
                        self._d_motion_instruct.va = self.va
                        print("#4.2")
                    #目標点がカメラ画角より右側にある場合
                    elif target_x > 420:
                        self._d_motion_instruct.va = -1 * self.va
                        print("#4.3")
                    else:
                        self._d_motion_instruct.va = 0.0
                        print("#4.4")
            color_img_flag = True


        '''
        深度画像に関する処理
        '''
        if self._depth_dataIn.isNew():
            print("#5")
            self._d_depth_data = self._depth_dataIn.read()
            print("#5.1")
            #深度データのByte列
            received_depth = self._d_depth_data.pixels
            print("#5.2")
            #Decode
            dept_json = json.loads(received_depth)
            print("#5.3")
            dept_dec = base64.b64decode(dept_json)
            print("#5.4")
            dept_np = np.frombuffer(dept_dec, dtype='uint8')
            print("#5.5")
            depth = cv2.imdecode(dept_np, flags=cv2.IMREAD_GRAYSCALE)
            print("#5.6")
            depth_scale = 256*depth
            print("#5.7")
            #カラー画像にする
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_scale, alpha=0.03), cv2.COLORMAP_JET)
            print("#6")
            #対象までの距離
            try:
                target_dist = depth_scale[target_y][target_x] / 1000
                print("#6.1")
            #対象が検出出来なかった場合の処理
            except:
                target_dist = 0
                target_x = int(self.width / 2)
                target_y = int(self.height / 2)
                print("#6.2")
            #対象の位置を円で囲う
            cv2.circle(depth_colormap, (target_x, target_y), 10, (255, 255, 255), thickness=2)
            cv2.putText(depth_colormap, str(target_dist), (5, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
            print("#6.3")
            depth_img_flag = True

            '''
            ロボットへの移動指令
            '''
            if person_flag:
                #対象まで充分な距離がある
                if target_dist > self.slow_dist[0]:
                    self._d_motion_instruct.vx = self.normal_speed

                #対象に少し近い場合
                elif (target_dist < self.slow_dist[0]) and (target_dist > self.slow_dist[1]):
                    self._d_motion_instruct.vx = self.slow_speeds[0]

                #更に近い場合
                elif (target_dist < self.slow_dist[1]) and (target_dist > self.min_dist):
                    self._d_motion_instruct.vx = self.slow_speeds[1]

                #近づきすぎた場合
                elif target_dist < self.min_dist:
                    self._d_motion_instruct.vx = 0.0

                print("#7")

        '''
        画像出力
        '''
        if color_img_flag and depth_img_flag:
            key_image_dim = keyimage.shape
            depth_map_dim = depth_colormap.shape

            if key_image_dim != depth_map_dim:
                resized_keyimage = cv2.resize(keyimage, dsize=(depth_map_dim[1], depth_map_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_keyimage, depth_colormap))

            else:
                images = np.htack((keyimage, depth_colormap))

            cv2.imshow("Image", images)
            cv2.waitKey(1)

            self.writer.write(images)


        self._motion_instructionOut.write()



        return RTC.RTC_OK
	
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

