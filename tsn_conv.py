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
from std_msgs.msg import String, Header
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

## do I need this?
#import caffe
from caffe.io import oversample
#from utils.io import flow_stack_oversample, fast_list2arr

class MyCaffeNet(CaffeNet):
    def __init__(self,*args, **kwargs):
        super(MyCaffeNet,self).__init__(*args, **kwargs) ## will it import things as well? if it doesn't then I will change the original file and rebuild the whole docker...

    ### the changes in these 2 are minuscule. I just want to output not the result from forward, but forward up to a point.
    def predict_single_frame(self, frame, score_name, over_sample=True, multiscale=None, frame_size=None, end=None):

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]

        if over_sample:
            if multiscale is None:
                os_frame = oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
            else:
                os_frame = []
                for scale in multiscale:
                    resized_frame = [cv2.resize(x, (0,0), fx=1.0/scale, fy=1.0/scale) for x in frame]
                    os_frame.extend(oversample(resized_frame, (self._sample_shape[2], self._sample_shape[3])))
        else:
            os_frame = fast_list2arr(frame)
        data = fast_list2arr([self._transformer.preprocess('data', x) for x in os_frame])

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data, end=end)
        return out[score_name].copy()

    def predict_single_flow_stack(self, frame, score_name, over_sample=True, frame_size=None, end=None):

        if frame_size is not None:
            frame = fast_list2arr([cv2.resize(x, frame_size) for x in frame])
        else:
            frame = fast_list2arr(frame)

        if over_sample:
            os_frame = flow_stack_oversample(frame, (self._sample_shape[2], self._sample_shape[3]))
        else:
            os_frame = fast_list2arr([frame])

        data = os_frame - np.float32(128.0)

        self._net.blobs['data'].reshape(*data.shape)
        self._net.reshape()
        out = self._net.forward(blobs=[score_name,], data=data, end=end)
        return out[score_name].copy()


class tsn_classifier:
  def __init__(self):
    global mypath
    # services provided
    self.reconfig_srv_ = rospy.Service('reconf_split',split, self.reconfig_srv)
    self.start_vidscores = rospy.Service('start_vidscores', Empty, self.start_vidscores)
    self.stop_vidscores = rospy.Service('stop_vidscores', Empty, self.stop_vidscores)
    # topics published
    self.scores_pub = rospy.Publisher("scores",caffe_tsn_ros.msg.Scores, queue_size=1)
    # self.label_fw_pub = rospy.Publisher("action_fw", String, queue_size=1)
    # self.label_pub = rospy.Publisher("action", String, queue_size=1)
    # self.ownlabel_pub = rospy.Publisher("action_own", String, queue_size=1)

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

    ###probably should use the nice rosparam thingy here to avoid these problems...
    self.framesize_width = rospy.get_param('~framesize_width',340)
    self.framesize_height = rospy.get_param('~framesize_height',256)

    # topics subscribed
    self.image_sub = rospy.Subscriber('video_topic', Image,self.callback,queue_size=1)

    # internals
    self.bridge = CvBridge()

    self.prototxt = mypath+'/models/'+ self.dataset +'/tsn_bn_inception_'+self.rgbOrFlow+'_deploy.prototxt'
    self.caffemodel = mypath+'/models/'+ self.dataset +'_split_'+str(self.split)+'_tsn_'+self.rgbOrFlow+'_reference_bn_inception.caffemodel'
    rospy.loginfo("loading prototxt {}".format(self.prototxt))
    rospy.loginfo("loading caffemodel {}".format(self.caffemodel))
    self.net = MyCaffeNet(self.prototxt, self.caffemodel, self.device_id)

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
              #self.ownlabel_pub.pub(self.ownvidscores)
              pass
          else:
              rospy.logerr('ownvidscores is empty!!!!!!!!!!!!!!! are we locking for too long?')
          #self.ownvidscores = []
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
        scoremsg = caffe_tsn_ros.msg.Scores()
        scoremsg.scores = scores
        self.scores_pub.publish(scoremsg)
        #print((scores))
        #print(np.shape(scores))

        if isinstance(scores, np.ndarray):
            #this publishes the instant time version, aka, per frame
            #self.label_pub.pub([scores])
            #rospy.logdebug("published the label for instant time version!")

            with self.lock:
                if self.startedownvid:
                    #self.ownvidscores.append(scores)
                    pass
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
