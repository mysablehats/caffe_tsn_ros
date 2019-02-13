#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('caffe_tsn_ros')
import sys
import rospy
import cv2
from caffe_tsn_ros.srv import *
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
np.set_printoptions(precision=2)
#import matplotlib.pyplot as plt
import itertools
import os
print(os.getcwd())
mypath = "/temporal-segment-networks"
sys.path.append(mypath)
sys.path.append(os.path.abspath(mypath+"/tools"))
sys.path.insert(0, mypath+'/lib/caffe-action/python') ## should use os.joinpath ...
from pyActionRecog.action_caffe import CaffeNet
import argparse
import math
import multiprocessing

class tsn_classifier:
  def __init__(self):
    global mypath
    self.reconfig_srv_ = rospy.Service('reconf_split',split, self.reconfig_srv)
    self.image_pub = rospy.Publisher("class_overlay_image_raw",Image, queue_size=1)
    self.label_fw_pub = rospy.Publisher("action_fw", String, queue_size=1)
    self.label_pub = rospy.Publisher("action", String, queue_size=1)
    self.dataset = rospy.get_param('~dataset','hmdb51')
    self.device_id = rospy.get_param('~device_id',0)
    self.split = rospy.get_param('~split',1)
    self.videotopic = rospy.get_param('~video_topic','videofiles/image_raw')
    self.classwindow = rospy.get_param('~classification_frame_window',50)
    self.actionlist = rospy.get_param('action_list', ['brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs','dive','draw_sword','dribble','drink','eat','fall_floor','fencing','flic_flac','golf','handstand','hit','hug','jump','kick','kick_ball','kiss','laugh','pick','pour','pullup','punch','push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball','shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault','stand','swing_baseball','sword','sword_exercise','talk','throw','turn','walk','wave'])
    self.framesize_width = rospy.get_param('~framesize_width',340)
    self.framesize_height = rospy.get_param('~framesize_height',256)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(self.videotopic, Image,self.callback,queue_size=1)
    from pyActionRecog.utils.video_funcs import default_aggregation_func
    self.defprox = default_aggregation_func
    self.frame_scores = []
    self.prototxt = mypath+'/models/'+ self.dataset +'/tsn_bn_inception_rgb_deploy.prototxt'
    self.caffemodel = mypath+'/models/'+ self.dataset +'_split_'+str(self.split)+'_tsn_rgb_reference_bn_inception.caffemodel'
    self.net = CaffeNet(self.prototxt, self.caffemodel, self.device_id)
    self.font = cv2.FONT_HERSHEY_SIMPLEX
    #print('hio')
    rospy.loginfo("waiting for callback from " + self.videotopic +" to do anything")

  def callback(self,data):
    rospy.logdebug("reached callback. that means I can read the Subscriber!")
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    #(rows,cols,channels) = cv_image.shape
    #if cols > 60 and rows > 60 :
      #cv2.circle(cv_image, (50,50), 10, 255)
    #TODO: frame_size should be a parameter
    scores = self.net.predict_single_frame([cv_image,], 'fc-action', frame_size=(self.framesize_width, self.framesize_height))
    #print((scores))

    #this publishes the instant time version, aka, per frame
    self.label_fw_pub.publish(String(self.actionlist[np.argmax(self.defprox([scores]]))]))
    rospy.logdebug("published the label for instant time version!")

    #this part publishes the frame_window version
    self.frame_scores.append(scores)
    if len(self.frame_scores)>self.classwindow:
        curract_fw = self.actionlist[np.argmax(self.defprox(self.frame_scores))]
        cv2.putText(cv_image,curract_fw,(10,226), self.font, 1,(255,255,255),2)
        self.frame_scores.pop(0)
        self.label_fw_pub.publish(String(curract_fw))
	rospy.logdebug("published the label for the frame window version!")


    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      rospy.logdebug("published image")
    except CvBridgeError as e:
      print(e)

  def reconfig_srv(self, req):
      self.image_sub.unregister()
      self.split = req.Split
      rospy.loginfo("reading split:"+str(req.Split))
      #print(req.Split)
      self.caffemodel = mypath+'/models/'+ self.dataset +'_split_'+str(self.split)+'_tsn_rgb_reference_bn_inception.caffemodel'
      self.net = CaffeNet(self.prototxt, self.caffemodel, self.device_id)
      self.image_sub = rospy.Subscriber(self.videotopic, Image,self.callback,queue_size=1)
      #print('Dum')
      return []

def main(argss):
  rospy.init_node('action_classifier', anonymous=True)
  ic = tsn_classifier()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
