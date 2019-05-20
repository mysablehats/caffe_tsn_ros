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
from std_msgs.msg import String, Header, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
np.set_printoptions(precision=2)
#import matplotlib.pyplot as plt
import itertools
import os
### I should figure out the pythonic way of doing thisq
print(os.getcwd())
mypath = "/temporal-segment-networks"
sys.path.append(mypath)
sys.path.append(os.path.abspath(mypath+"/tools"))
sys.path.insert(0, mypath+'/lib/caffe-action/python') ## should use os.path.join?
from pyActionRecog.action_caffe import CaffeNet, fast_list2arr, flow_stack_oversample
import argparse
import math
import multiprocessing
import caffe_tsn_ros.msg
from copy import deepcopy

class tsn_classifier:
  def __init__(self):
    global mypath
    # services provided
    self.reconfig_srv_ = rospy.Service('reconf_split',split, self.reconfig_srv)
    self.start_vidscores = rospy.Service('start_vidscores', Empty, self.start_vidscores)
    self.stop_vidscores = rospy.Service('stop_vidscores', Empty, self.stop_vidscores)
    # topics published
    self.scores_pub = rospy.Publisher("scores",Float32MultiArray, queue_size=1)
    self.score_fw_pub = rospy.Publisher("action_fw", caffe_tsn_ros.msg.ScoreArray, queue_size=1)
    # self.label_pub = rospy.Publisher("action", String, queue_size=1)
    self.ownlabel_pub = rospy.Publisher("action_own", caffe_tsn_ros.msg.ScoreArray, queue_size=1)

    # parameters
    self.dataset = rospy.get_param('~dataset','hmdb51')
    self.device_id = rospy.get_param('~device_id',0)
    self.split = rospy.get_param('~split',1)
    self.rgbOrFlow = rospy.get_param('~classifier_type')
    self.step = rospy.get_param('~step',6)
    # this should actually be
    # step = (frame_cnt - stack_depth) / (args.num_frame_per_video-1)
    # it will change depending on the action length, a value I don't have if I am classifying real time, but that I could get if I am doing it by service calls!

    # stack_depth is 1 for rgb and 5 for flows. I am letting it be 5 to test creating an array of cv_images
    self.stack_depth = rospy.get_param('~stack_depth',5)
    self.stack_count = 0
    self.cv_image_stack = []

    self.classwindow = rospy.get_param('~classification_frame_window',50)

    ###probably should use the nice rosparam thingy here to avoid these problems...
    self.framesize_width = rospy.get_param('~framesize_width',340)
    self.framesize_height = rospy.get_param('~framesize_height',256)

    # topics subscribed
    self.image_sub = rospy.Subscriber('video_topic', Image, self.callback,queue_size=1)

    # internals
    self.bridge = CvBridge()

    self.prototxt = mypath+'/models/'+ self.dataset +'/tsn_bn_inception_'+self.rgbOrFlow+'_deploy.prototxt'
    self.caffemodel = mypath+'/models/'+ self.dataset +'_split_'+str(self.split)+'_tsn_'+self.rgbOrFlow+'_reference_bn_inception.caffemodel'
    rospy.loginfo("loading prototxt {}".format(self.prototxt))
    rospy.loginfo("loading caffemodel {}".format(self.caffemodel))
    self.net = CaffeNet(self.prototxt, self.caffemodel, self.device_id)

    self.frame_scores = caffe_tsn_ros.msg.ScoreArray()
    self.ownvidscores = caffe_tsn_ros.msg.ScoreArray()
    # when I instantiate the classifier, the startedownvid is working already. this influences how vsmf_srv will behave, so it needs to be like this, I think.
    self.startedownvid = True
    self.lock = threading.Lock()

    #publishers
    ###need to publish the not-last layer thingy!

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
              #### I am going to publish now a set of matrices, right?
              self.ownlabel_pub.pub(self.ownvidscores)
              pass
          else:
              rospy.logerr('ownvidscores is empty!!!!!!!!!!!!!!! are we locking for too long?')
          self.ownvidscores = caffe_tsn_ros.msg.ScoreArray()
          rospy.logdebug("published the label for the own video version!")
          rospy.logwarn("stopped classifying own vid")

      return []
  def callback(self,data):
    rospy.logdebug("reached callback. that means I can read the Subscriber!")
    rospy.set_param('~alive',1)
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      #print(e)
      rospy.logerr(e)

    #since I am not using stacks for rgb images, I can prevent from making the rgb version any slower by using an if statement here
    if self.rgbOrFlow == 'flow':
        ## and I want the combined flow version here, don't I? so I need to strip the frame apart into components. I think it is better than
        #self.cv_image_stack.append(cv_image)
        #this would be incorrect, i need to get each matrix and put it together in a stack. first x then y
        # from ros_flow_bg, I can see that x is the 0 component and y the 1 component. I hope bgr8 stays bgr8 and we don't flip things around here!
        self.cv_image_stack.append(cv_image[:,:,0])
        self.cv_image_stack.append(cv_image[:,:,1])
#        self.cv_image_stack.extend([cv_image[:,:,0], cv_image[:,:,1]])


        if len(self.cv_image_stack)>2*self.stack_depth: #keeps at most 2*self.stack_depth images in cv_image_stack, the 2 comes from the fact that we are using flow_x and flow_y
            self.cv_image_stack.pop(0)
            self.cv_image_stack.pop(0)
    if self.stack_count%self.step == 0:
        rospy.logdebug("reached execution part of callback!")
        self.stack_count = 0 ## we don't keep a large number here.
        scores = None
        ### i can maybe use a lambda to abstract this. is it faster than an if though?
        if self.rgbOrFlow == 'rgb':
            scores = self.net.predict_single_frame([cv_image,], 'global_pool', frame_size=(self.framesize_width, self.framesize_height))
        elif self.rgbOrFlow == 'flow' and len(self.cv_image_stack)==10:
            scores = self.net.predict_single_flow_stack(self.cv_image_stack, 'global_pool', frame_size=(self.framesize_width, self.framesize_height))

        #print(type(scores))
        #print(scores.dtype)
        #scoremsg = caffe_tsn_ros.msg.Scores()
        #scoremsg.test = 'hello'
        #scoremsg.scores = scores
        #print(scoremsg)
        #scores = np.squeeze(scores)
        #scores = np.array([[[1.,2.],[3.,4.],[5.,6.]],[[11.,12.],[13.,14.],[15.,16.]]],dtype='float32')
        #self.scores_pub.publish(self.bridge.cv2_to_imgmsg(scores, '32FC1'))
        #self.scores_pub.publish(scoremsg)
        #print(self.scores_pub.get_num_connections())
        #print((scores))
        #print(np.shape(scores))
        #print(scores.shape)
        if isinstance(scores, np.ndarray):
            #this publishes the instant time version, aka, per frame
            #self.label_pub.pub([scores])
            ## Not sure if this MultiArrayDimension thing is working. In any
            ## case, it is already a LOT of data per frame; publishing the chunk
            ## at the end of the video, all at the same time would probably
            ## cause a lot of unnecessary waiting
            # TO CONSIDER: this is fast, so I think it doesn't matter, but I
            # believe the layout could be pre-set, since it doesn't change on a
            # frame by frame basis.
            scoresmsg = Float32MultiArray()
            scoresmsg.layout.dim = []
            dims = np.array(scores.shape)
            scoresize = dims.prod()/float(scores.nbytes)
            for i in range(0,len(dims)):
                #print(i)
                scoresmsg.layout.dim.append(MultiArrayDimension())
                scoresmsg.layout.dim[i].size = dims[i]
                scoresmsg.layout.dim[i].stride = dims[i:].prod()/scoresize
                scoresmsg.layout.dim[i].label = 'dim_%d'%i
                #print(scoresmsg.layout.dim[i].size)
                #print(scoresmsg.layout.dim[i].stride)
            scoresmsg.data = np.frombuffer(scores.tobytes(),'float32')
            self.scores_pub.publish(scoresmsg)
            #rospy.logdebug("published the label for instant time version!")

            #this part publishes the frame_window version
            self.frame_scores.scores.append(scores)
            if len(self.frame_scores.scores)>self.classwindow:
                self.frame_scores.scores.pop(0)
                self.label_fw_pub.pub(self.frame_scores)
                rospy.logdebug("published the label for the frame window version!")

            with self.lock:
                if self.startedownvid:
                    self.ownvidscores.scores.append(scores)
                    #pass
                else:
                    rospy.logdebug_throttle(20,"waiting for start_vidscores call to start classifying ownvid")

    self.stack_count = self.stack_count + 1
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
      self.caffemodel = mypath+'/models/'+ self.dataset +'_split_'+str(self.split)+'_tsn_'+self.rgbOrFlow+'_reference_bn_inception.caffemodel'
      self.net = CaffeNet(self.prototxt, self.caffemodel, self.device_id)
      self.image_sub = rospy.Subscriber('video_topic', Image,self.callback,queue_size=1)
      #print('Dum')
      return []

def main(argss):
  rospy.init_node('action_classifier', anonymous=True, log_level=rospy.INFO)
  ic = tsn_classifier()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

  except rospy.ROSException as e:
    rospy.spin()
    rospy.logerr(e)


if __name__ == '__main__':
    main(sys.argv)
