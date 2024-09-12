

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python -*-

"""
 @file RealSense.py
 @brief Get RGB image and depth imformation from RealSense
 @date $Date$


"""
import sys
import time
sys.path.append(".")

# Import RTM module
import RTC
import OpenRTM_aist

import pyrealsense2 as rs
import numpy as np
import cv2 

import base64
import json

# Import Service implementation class
# <rtc-template block="service_impl">

# </rtc-template>

# Import Service stub modules
# <rtc-template block="consumer_import">
# </rtc-template>


# This module's spesification
# <rtc-template block="module_spec">
realsense_spec = ["implementation_id", "RealSense", 
		 "type_name",         "RealSense", 
		 "description",       "Get RGB image and depth imformation from RealSense", 
		 "version",           "1.0.0", 
		 "vendor",            "Kashima", 
		 "category",          "Camer", 
		 "activity_type",     "STATIC", 
		 "max_instance",      "1", 
		 "language",          "Python", 
		 "lang_type",         "SCRIPT",
		 ""]
# </rtc-template>
#person detector 
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
hogParams = {}

##
# @class RealSense
# @brief Get RGB image and depth imformation from RealSense
# 
# 
class RealSense(OpenRTM_aist.DataFlowComponentBase):
	
	##
	# @brief constructor
	# @param manager Maneger Object
	# 
	def __init__(self, manager):
		OpenRTM_aist.DataFlowComponentBase.__init__(self, manager)

		#
		# 
		# rgb_img_arg = [None] * ((len(Img._d_CameraImage) - 4) / 2)
		self._d_rgb_img = RTC.CameraImage(RTC.Time(0,0),0,0,0,"",0.0,"")
		
		"""
		"""
		self._RGB_imageOut = OpenRTM_aist.OutPort("RGB_image", self._d_rgb_img)
		Depth_arg = [None] * ((len(RTC._d_PointCloud) - 4) / 2)
		self._d_Depth = RTC.CameraImage(RTC.Time(0,0),0,0,0,"",0.0,"")
		"""
		"""
		self._DepthOut = OpenRTM_aist.OutPort("Depth", self._d_Depth)		

		# initialize of configuration-data.
		self._conf_width = [640]
		self._conf_height = [480]
		self._conf_bpp = [24]

		self.fps = None
		self.width = None
		self.height = None
		# <rtc-template block="init_conf_param">
		
		# </rtc-template>


		 
	##
	#
	# The initialize action (on CREATED->ALIVE transition)
	# formaer rtc_init_entry() 
	# 
	# @return RTC::ReturnCode_t
	# 
	#
	def onInitialize(self):
		# Bind variables and configuration variable
		self.bindParameter("Width", self._conf_width, 640)
		self.bindParameter("Height", self._conf_height, 480)
		self.bindParameter("Bits_Per_Pixel", self._conf_bpp, 24)
		
		# Set InPort buffers
		
		# Set OutPort buffers
		self.addOutPort("RGB_image",self._RGB_imageOut)
		self.addOutPort("Depth",self._DepthOut)
		
		# Set service provider to Ports
		
		# Set service consumers to Ports
		
		# Set CORBA Service Ports
		
		return RTC.RTC_OK
	
	#	##
	#	# 
	#	# The finalize action (on ALIVE->END transition)
	#	# formaer rtc_exiting_entry()
	#	# 
	#	# @return RTC::ReturnCode_t
	#
	#	# 
	#def onFinalize(self):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The startup action when ExecutionContext startup
	#	# former rtc_starting_entry()
	#	# 
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onStartup(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The shutdown action when ExecutionContext stop
	#	# former rtc_stopping_entry()
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onShutdown(self, ec_id):
	#
	#	return RTC.RTC_OK
	
		##
		#
		# The activated action (Active state entry action)
		# former rtc_active_entry()
		#
		# @param ec_id target ExecutionContext Id
		# 
		# @return RTC::ReturnCode_t
		#
		#
	def onActivated(self, ec_id):
		print("Activate RealSense")
		print("Preparing RealSense...")

		#Configure depth and color streams
		self.pipeline = rs.pipeline()
		self.config = rs.config()

		#Get device produc line for setting a supporting resolution
		pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
		pipeline_profile = self.config.resolve(pipeline_wrapper)
		device = pipeline_profile.get_device()
		device_product_line = str(device.get_info(rs.camera_info.product_line))
		found_rgb = False

		for s in device.sensors:
			if s.get_info(rs.camera_info.name) == 'RGB Camera':
				found_rgb = True

				break

		if not found_rgb:
			print("The demo requires Depth camera with Color sensor")
			exit(0)

		self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
		
		if device_product_line == 'L500':
			self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)

		else:
			self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)


		#Start streaming
		self.pipeline.start(self.config)
		self.start_time = time.time()
		self.frame_num = 1

		print("RealSense ready")

		return RTC.RTC_OK
	
		##
		#
		# The deactivated action (Active state exit action)
		# former rtc_active_exit()
		#
		# @param ec_id target ExecutionContext Id
		#
		# @return RTC::ReturnCode_t
		#
		#
	def onDeactivated(self, ec_id):
		print("Deactivete RealSense")
		#Stop streaming
		self.pipeline.stop()
		cv2.destroyAllWindows()

		return RTC.RTC_OK
	
		##
		#
		# The execution action that is invoked periodically
		# former rtc_active_do()
		#
		# @param ec_id target ExecutionContext Id
		#
		# @return RTC::ReturnCode_t
		#
		#
	def onExecute(self, ec_id):
		#Wait for a coherent pair of frames: depth and color
		frames = self.pipeline.wait_for_frames()
		depth_frame = frames.get_depth_frame()
		color_frame = frames.get_color_frame()

		if not depth_frame:
			print("Depth frame does not exist")

		elif not color_frame:
			print("Color frame does not exist")		

		#Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())
		#print("#1")
		#Apply colormap on depth image (image must be converted to 8-bit per pixel first)
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

		#depth_colormap_dim = depth_colormap.shape
		#color_colormap_dim = color_image.shape
		'''
		if depth_colormap_dim != color_colormap_dim:
			resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[1]), interpolation=cv2.INTER_AREA)
			images = np.hstack((resized_color_image, depth_colormap))

		else:
			images = np.hstack((color_image, depth_colormap))
		'''
		#print("#3")
		
		#before_tobytes = time.time()
		'''
		Encode RGB image
		'''
		_, encimg = cv2.imencode(".png", color_image)
		img_str = encimg.tobytes()
		img_byte = base64.b64encode(img_str).decode("utf-8")
		rgb_data = json.dumps(img_byte).encode("utf-8")		
		#print("#2")
		self._d_rgb_img.width = 640
		self._d_rgb_img.height = 480
		self._d_rgb_img.format = "png"
		
		self._d_rgb_img.pixels = rgb_data
		#self._d_rgb_img.pixels = "".join([chr(value) for value in color_image.flat])
		#print("#3.1")

		'''
		encode depth frame
		'''
		_, encdept = cv2.imencode(".png", depth_image) 
		dept_str = encdept.tobytes()
		dept_byte = base64.b64encode(dept_str).decode("utf-8")
		dept_data = json.dumps(dept_byte).encode("utf-8")
		#dept_data = depth_image.flatten().tobytes()
		self._d_Depth.width = 480
		self._d_Depth.height = 640
		self._d_Depth.format = "png"

		self._d_Depth.pixels = dept_data
		#print("center dist >", depth_frame.get_distance(320, 240))
		
		#print("depth image > ", depth_image)
		#(480, 640)
		#print("len > ", len(self._d_Depth.pixels))
		#print("color map > ", depth_colormap.shape)		

		#print("#3")
		#before_write = time.time()
		self._RGB_imageOut.write()
		#print("#3.1")
		self._DepthOut.write()
		#after_write = time.time()
		#print("time to write > ", after_write - before_write)
		#print("#4")
		#dist = depth_frame.get_distance(320, 240)
		#print("Dist > ", dist)

		'''
		dept_json = json.loads(dept_data)
		dept_dec = base64.b64decode(dept_json)
		dept_np = np.frombuffer(dept_dec, dtype='uint8')
		depth = cv2.imdecode(dept_np, flags=cv2.IMREAD_GRAYSCALE)
		depthmap_dec = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
		cv2.imshow("Depth", depth_colormap)
		cv2.waitKey(1)
		'''

		
		return RTC.RTC_OK
	
	#	##
	#	#
	#	# The aborting action when main logic error occurred.
	#	# former rtc_aborting_entry()
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onAborting(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The error action in ERROR state
	#	# former rtc_error_do()
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onError(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The reset action that is invoked resetting
	#	# This is same but different the former rtc_init_entry()
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onReset(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The state update action that is invoked after onExecute() action
	#	# no corresponding operation exists in OpenRTm-aist-0.2.0
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#

	#	#
	#def onStateUpdate(self, ec_id):
	#
	#	return RTC.RTC_OK
	
	#	##
	#	#
	#	# The action that is invoked when execution context's rate is changed
	#	# no corresponding operation exists in OpenRTm-aist-0.2.0
	#	#
	#	# @param ec_id target ExecutionContext Id
	#	#
	#	# @return RTC::ReturnCode_t
	#	#
	#	#
	#def onRateChanged(self, ec_id):
	#
	#	return RTC.RTC_OK
	



def RealSenseInit(manager):
    profile = OpenRTM_aist.Properties(defaults_str=realsense_spec)
    manager.registerFactory(profile,
                            RealSense,
                            OpenRTM_aist.Delete)

def MyModuleInit(manager):
    RealSenseInit(manager)

    # Create a component
    comp = manager.createComponent("RealSense")

def main():
	mgr = OpenRTM_aist.Manager.init(sys.argv)
	mgr.setModuleInitProc(MyModuleInit)
	mgr.activateManager()
	mgr.runManager()

if __name__ == "__main__":
	main()

