#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('caffe_tsn_ros')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import numpy as np
np.set_printoptions(precision=2)
#import matplotlib.pyplot as plt
import itertools
import os
print(os.getcwd())
sys.path.append(os.path.abspath("/temporal-segment-networks/tools"))
sys.path.append('/temporal-segment-networks')
sys.path.insert(0, '/temporal-segment-networks/lib/caffe-action/python')
import argparse
import math
import multiprocessing

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_raw",Image)
    self.actionlist = ['brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs','dive','draw_sword','dribble','drink','eat','fall_floor','fencing','flic_flac','golf','handstand','hit','hug','jump','kick','kick_ball','kiss','laugh','pick','pour','pullup','punch','push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball','shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault','stand','swing_baseball','sword','sword_exercise','talk','throw','turn','walk','wave']
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("videofiles/image_raw", Image,self.callback,queue_size=1)
    from pyActionRecog.utils.video_funcs import default_aggregation_func
    from pyActionRecog.action_caffe import CaffeNet
    self.defprox = default_aggregation_func
    self.frame_scores = []
    self.net = CaffeNet('models/hmdb51/tsn_bn_inception_rgb_deploy.prototxt', 'models/hmdb51_split_1_tsn_rgb_reference_bn_inception.caffemodel', 0)
    self.font = cv2.FONT_HERSHEY_SIMPLEX

  def callback(self,data):
    #rospy.loginfo_once("reached callback. that means I can read the Subscriber!")
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    if cols > 60 and rows > 60 :
      cv2.circle(cv_image, (50,50), 10, 255)
      scores = self.net.predict_single_frame([cv_image,], 'fc-action', frame_size=(340, 256))
      #print(scores)
      self.frame_scores.append(scores)
    if len(self.frame_scores)>50:
        curract = self.actionlist[np.argmax(self.defprox(self.frame_scores))]
        cv2.putText(cv_image,curract,(10,246), self.font, 2,(255,255,255),2)
        self.frame_scores = []
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(argss):
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
