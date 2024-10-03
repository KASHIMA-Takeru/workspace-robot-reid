#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python -*-

# <rtc-template block="description">
"""
 @file MakeGallery.py
 @brief make gallery images
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

import json
import base64
import numpy as np
import cv2


import datetime
import os
import os.path as osp

#current directory: C:\Users\ab19109\workspce_robot_reid\MakeGallery

from MyTools import openpose_processor as opp



# Import Service implementation class
# <rtc-template block="service_impl">

# </rtc-template>

# Import Service stub modules
# <rtc-template block="consumer_import">
# </rtc-template>


# This module's spesification
# <rtc-template block="module_spec">
makegallery_spec = ["implementation_id", "MakeGallery", 
         "type_name",         "MakeGallery", 
         "description",       "make gallery images", 
         "version",           "1.0.0", 
         "vendor",            "Kashima", 
         "category",          "Camera", 
         "activity_type",     "STATIC", 
         "max_instance",      "1", 
         "language",          "Python", 
         "lang_type",         "SCRIPT",
         ""]
# </rtc-template>

# <rtc-template block="component_description">
##
# @class MakeGallery
# @brief make gallery images
# 
# 
# </rtc-template>
class MakeGallery(OpenRTM_aist.DataFlowComponentBase):
	
    ##
    # @brief constructor
    # @param manager Maneger Object
    # 
    def __init__(self, manager):
        OpenRTM_aist.DataFlowComponentBase.__init__(self, manager)

        self._d_image = RTC.CameraImage(RTC.Time(0, 0), 0, 0, 0, "", 0.0, "")
        """
        """
        self._imageIn = OpenRTM_aist.InPort("image", self._d_image)


		


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
        self.addInPort("image",self._imageIn)
		
        # Set OutPort buffers
		
        # Set service provider to Ports
		
        # Set service consumers to Ports
		
        # Set CORBA Service Ports

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
        date = datetime.datetime.now().strftime("%m%d")
        print("Start")
        self.pid = str(input("Input Person ID > ")).zfill(3)
        self.camid = str(input("Input Camera ID > ")).zfill(2)
        self.save_dir = r'D:\master_research\Robot\gallery_storage\{}'.format(date)
        self.part_save_dir = osp.join(self.save_dir, 'part')

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.part_save_dir, exist_ok=True)

        print("Ready")
    
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
        print("Finish")
        cv2.destroyAllWindows()
    
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
        if self._imageIn.isNew():
            self._d_image = self._imageIn.read()
            received_data = self._d_image.pixels
            #print("#1")
            #デコード            
            data_json = json.loads(received_data)
            image_dec = base64.b64decode(data_json)
            data_np = np.frombuffer(image_dec, dtype='uint8')
            image = cv2.imdecode(data_np, 1)
            #print("#2")

            keypoints, keyimage = opp.detect_keypoints(image)
            
            cv2.imshow("image", keyimage)
            cv2.waitKey(1)

            #print("keypoints > ", keypoints)
            #人物画像作成
            if type(keypoints) == np.ndarray:
                bbox_list, _ = opp.make_person_image(image, keypoints=keypoints)
                for key in keypoints:
                    cropped_images = opp.make_part_image(image, keypoints=key)
                #print("#4")    
            
                
                for i, bbox in enumerate(bbox_list):
                    #print("person > ", type(person))
                    #保存名
                    timestamp = datetime.datetime.now().strftime("%H%M_%S")
                    save_name = '{}_s{}_{}_{}'.format(self.pid, self.camid, timestamp, str(i))
                    #print("person name > ", save_name)
                    #print("save > ", save_name)
                    #画像保存          
                    person = image[bbox[0]: bbox[1], bbox[2]: bbox[3]]
                    cv2.imwrite(osp.join(self.save_dir, save_name + '.jpg'), person)
                    

                for part in cropped_images.keys():
                    try:
                        save_folder = osp.join(self.part_save_dir, part)
                        os.makedirs(save_folder, exist_ok=True)
                        part_save_name = save_name + '_{}'.format(part)
                        #print("name > ", part_save_name)
                        cv2.imwrite(osp.join(save_folder, part_save_name + '.jpg'), cropped_images[part])

                    except:
                        pass


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
	



def MakeGalleryInit(manager):
    profile = OpenRTM_aist.Properties(defaults_str=makegallery_spec)
    manager.registerFactory(profile,
                            MakeGallery,
                            OpenRTM_aist.Delete)

def MyModuleInit(manager):
    MakeGalleryInit(manager)

    # create instance_name option for createComponent()
    instance_name = [i for i in sys.argv if "--instance_name=" in i]
    if instance_name:
        args = instance_name[0].replace("--", "?")
    else:
        args = ""
  
    # Create a component
    comp = manager.createComponent("MakeGallery" + args)

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

