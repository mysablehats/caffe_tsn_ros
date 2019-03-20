#!/usr/bin/env python
from __future__ import print_function

import threading

import roslib
roslib.load_manifest('caffe_tsn_ros')
import sys
import rospy
import rosparam
import cv2
from std_srvs.srv import Empty
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
sys.path.insert(0, mypath+'/lib/caffe-action/python') ## should use os.joinpath?
from pyActionRecog.action_caffe import CaffeNet
import argparse
import math
import multiprocessing
import caffe_tsn_ros.msg
from copy import deepcopy

class FunnyPublisher:
    def __init__(self,name, actionlist, defprox):
        self.label_pub = rospy.Publisher(name+'_label',String, queue_size=1)
        self.array_pub = rospy.Publisher(name+'_label_dic',caffe_tsn_ros.msg.ActionDic, queue_size=1)
        self.actionlist = actionlist
        #rospy.logdebug('my action list:')
        #rospy.logdebug(self.actionlist)
        self.defprox = defprox
        self.name = name
        self.lastaction = []
        rospy.loginfo('FP {} initialized'.format(self.name))
    def pub(self,vidscores):
        try:
            #rospy.logdebug(len(vidscores))
            #rospy.logdebug(vidscores)
            conflist = self.defprox(vidscores)
            #rospy.logdebug('length of conflist %d () for hmdb51 should be 51'%len(conflist))
            #rospy.logdebug(conflist)
            self.lastaction = self.actionlist[np.argmax(conflist)]
            self.label_pub.publish(self.lastaction)
            actionDic_but_it_is_a_list = []
            for action,confidence in zip(self.actionlist,conflist):
                thisAction = caffe_tsn_ros.msg.Action()
                thisAction.action = action
                thisAction.confidence = confidence
                #rospy.logdebug('what I am stacking: action, confidence: (%s,%f)'%(action, confidence))
                actionDic_but_it_is_a_list.append(thisAction)
                thisAction = []
            #for actionvec in actionDic_but_it_is_a_list:

                #rospy.logdebug('what I actually stacked: action, confidence: (%s,%f)'%(actionvec.action, actionvec.confidence))
            #rospy.logdebug('what I was meant to publish is: len %d'%len(actionDic_but_it_is_a_list))
            #rospy.logdebug(actionDic_but_it_is_a_list)
            self.array_pub.publish(actionDic_but_it_is_a_list)
            rospy.logdebug('FP {} published alright. '.format(self.name))
        except Exception as e:
            rospy.logerr('FP {} publisher failed. '.format(self.name))
            rospy.logerr(e)
    def lastactionpublished(self):
        rospy.loginfo('Last action published was: %s'%self.lastaction)
class tsn_classifier:
  def __init__(self):
    global mypath
    # services provided
    self.reconfig_srv_ = rospy.Service('reconf_split',split, self.reconfig_srv)
    self.start_vidscores = rospy.Service('start_vidscores', Empty, self.start_vidscores)
    self.stop_vidscores = rospy.Service('stop_vidscores', Empty, self.stop_vidscores)
    # topics published
    # self.image_pub = rospy.Publisher("class_overlay_image_raw",Image, queue_size=1)
    # self.label_fw_pub = rospy.Publisher("action_fw", String, queue_size=1)
    # self.label_pub = rospy.Publisher("action", String, queue_size=1)
    # self.ownlabel_pub = rospy.Publisher("action_own", String, queue_size=1)

    # parameters
    self.dataset = rospy.get_param('~dataset','hmdb51')
    self.device_id = rospy.get_param('~device_id',0)
    self.split = rospy.get_param('~split',1)
    self.step = rospy.get_param('~step',6)
    # this should actually be
    # step = (frame_cnt - stack_depth) / (args.num_frame_per_video-1)
    # it will change depending on the action length, a value I don't have if I am classifying real time, but that I could get if I am doing it by service calls!

    self.stack_depth = rospy.get_param('~stack_depth',5)
    # stack_depth is 1 for rgb and 5 for flows. I am letting it be 5 to test creating an array of cv_images

    self.classwindow = rospy.get_param('~classification_frame_window',50)
    #whatswrong = (rospy.resolve_name('~action_list'))
    #rospy.spin()
    self.actionlist = rosparam.get_param(rospy.resolve_name('~action_list')) #"['brush_hair','cartwheel','catch','chew','clap','climb','climb_stairs','dive','draw_sword','dribble','drink','eat','fall_floor','fencing','flic_flac','golf','handstand','hit','hug','jump','kick','kick_ball','kiss','laugh','pick','pour','pullup','punch','push','pushup','ride_bike','ride_horse','run','shake_hands','shoot_ball','shoot_bow','shoot_gun','sit','situp','smile','smoke','somersault','stand','swing_baseball','sword','sword_exercise','talk','throw','turn','walk','wave']")
    #if type(self.actionlist) is str:
    #    self.actionlist = eval(self.actionlist)
    self.actionlist.sort()
    self.chooselist = rosparam.get_param(rospy.resolve_name('~choose_list')) ## I must be doing something wrong here for this name not to be resolved. maybe it is because each node here should probably have its own init_node and it doesn't
    #if type(self.chooselist) is str:
    #    self.chooselist = eval(self.chooselist)
    self.chooselist.sort()
    ###probably should use the nice rosparam thingy here to avoid these problems...
    self.framesize_width = rospy.get_param('~framesize_width',340)
    self.framesize_height = rospy.get_param('~framesize_height',256)

    # topics subscribed
    self.image_sub = rospy.Subscriber('video_topic', Image,self.callback,queue_size=1)

    # internals
    self.bridge = CvBridge()
    from pyActionRecog.utils.video_funcs import default_aggregation_func
    if self.chooselist:
        keepi = []
        rospy.logwarn('defined own subset of actions! classification will be reduced to smaller set of choices, namely:'+str(self.chooselist))
        #print(range(0,len(self.actionlist)))
        for i in range(0,len(self.actionlist)):
             for j in range(0, len(self.chooselist)):
                 #print(self.actionlist[i])
                 #print( self.chooselist[j])
                 if self.actionlist[i] == self.chooselist[j]:
                     keepi.append(i)
        tobedeleted = set(range(0,len(self.actionlist)))-set(keepi)
        #print(tobedeleted)
        self.defprox = lambda x: np.delete(default_aggregation_func(x),list(tobedeleted))
        self.actionlist = self.chooselist
    else:
        rospy.logwarn('No choose_list defined. Will classify within the whole set. ')
        self.defprox = default_aggregation_func
    self.frame_scores = []
    self.prototxt = mypath+'/models/'+ self.dataset +'/tsn_bn_inception_rgb_deploy.prototxt'
    self.caffemodel = mypath+'/models/'+ self.dataset +'_split_'+str(self.split)+'_tsn_rgb_reference_bn_inception.caffemodel'
    self.net = CaffeNet(self.prototxt, self.caffemodel, self.device_id)

    self.ownvidscores = []
    # when I instantiate the classifier, the startedownvid is working already. this influences how vsmf_srv will behave, so it needs to be like this, I think.
    self.startedownvid = True
    self.lock = threading.Lock()

    #publishers
    self.label_fw_pub = FunnyPublisher("action_fw", self.actionlist, self.defprox)
    self.label_pub = FunnyPublisher("action", self.actionlist, self.defprox)
    self.ownlabel_pub = FunnyPublisher("action_own", self.actionlist, self.defprox)
    rospy.set_param('~alive',0.5)
    rospy.loginfo("waiting for callback from " +rospy.resolve_name('video_topic') +" to do anything")

  def start_vidscores(self,req):
      # I will need to use locks here, I think...
      with self.lock:
          self.startedownvid = True
      rospy.logwarn("Started classifying own vid!")
      return []
  def stop_vidscores(self,req):
      # I will need to use locks here, I think...
      with self.lock:
          self.startedownvid = False
          if self.ownvidscores:
              self.ownlabel_pub.pub(self.ownvidscores)
          else:
              rospy.logerr('ownvidscores is empty!!!!!!!!!!!!!!! are we locking for too long?')
          self.ownvidscores = []
          rospy.logdebug("published the label for the own video version!")
          rospy.logwarn("stopped classifying own vid")

      return []
  def callback(self,data):
    rospy.logdebug("reached callback. that means I can read the Subscriber!")
    rospy.set_param('~alive',1)
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    scores = self.net.predict_single_frame([cv_image,], 'fc-action', frame_size=(self.framesize_width, self.framesize_height))
    #print((scores))

    #this publishes the instant time version, aka, per frame
    self.label_pub.pub([scores])
    rospy.logdebug("published the label for instant time version!")

    #this part publishes the frame_window version
    self.frame_scores.append(scores)
    if len(self.frame_scores)>self.classwindow:
        self.frame_scores.pop(0)
        self.label_fw_pub.pub(self.frame_scores)
        rospy.logdebug("published the label for the frame window version!")

    self.lock.acquire()
    if self.startedownvid:
        self.ownvidscores.append(scores)
    else:
        rospy.logdebug_throttle(20,"waiting for start_vidscores call to start classifying ownvid")
    self.lock.release()


    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    #   rospy.logdebug("published image")
    # except CvBridgeError as e:
    #   print(e)

  def reconfig_srv(self, req):
      # why not use standard ros reconfigure stds?
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
  rospy.init_node('action_classifier', anonymous=True, log_level=rospy.INFO)
  ic = tsn_classifier()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  # cv2.destroyAllWindows()
  except rospy.ROSException as e:
    rospy.spin()
    rospy.logerr(e)


if __name__ == '__main__':
    main(sys.argv)
