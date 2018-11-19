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

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("videofiles/image_raw", Image,self.callback,queue_size=1)
    self.frame_scores = []
    self.net = CaffeNet('models/hmdb51/tsn_bn_inception_rgb_deploy.prototxt', 'models/hmdb51_split_1_tsn_rgb_reference_bn_inception.caffemodel', 0)
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
      print(scores)
      self.frame_scores.append(scores)
    if len(self.frame_scores)>50:
        print([np.argmax(default_aggregation_func(x[0])) for x in video_scores_rgb])
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(argss):
  def build_net():
    global net
    my_id = multiprocessing.current_process()._identity[0] \
        if args.num_worker > 1 else 1
    if gpu_list is None:
        net = CaffeNet(args.net_proto, args.net_weights, my_id-1)
    else:
        net = CaffeNet(args.net_proto, args.net_weights, gpu_list[my_id - 1])

  def eval_video(video):
    global net
    #### each video change has to trigger this guy.
    label = video[1]
    vid = video[0]

    video_frame_path = f_info[0][vid]
    cnt_indexer = 1
    frame_cnt = f_info[cnt_indexer][vid]

    stack_depth = 1
    step = (frame_cnt - stack_depth) / (args.num_frame_per_video-1)
    if step > 0:
        frame_ticks = range(1, min((2 + step * (args.num_frame_per_video-1)), frame_cnt+1), step)
    else:
        frame_ticks = [1] * args.num_frame_per_video

    assert(len(frame_ticks) == args.num_frame_per_video)

    frame_scores = []
    rospy.loginfo("frame_ticks: %s",frame_ticks)
    for tick in frame_ticks:
        name = '{}{:05d}.jpg'.format(args.rgb_prefix, tick)
        frame = cv2.imread(os.path.join(video_frame_path, name), cv2.IMREAD_COLOR)
        #here is where I probably want to load things from the topic
        # each new frame triggers this guy.
        scores = net.predict_single_frame([frame,], score_name, frame_size=(340, 256))
        frame_scores.append(scores)
    print('video {} done'.format(vid))
    sys.stdin.flush()
    return np.array(frame_scores), label

  global args
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()

#####A LOT of those parameters will not be neccessary anymore, with the whole splitting into rgb and flow and with my decision to remove some of the multiprocessing stuff. anyway, so far will keep the mess...
  sys.argv = ['','hmdb51','1','rgb','/temporal-segment-networks/my_of/','models/hmdb51/tsn_bn_inception_rgb_deploy.prototxt',\
            'models/hmdb51_split_1_tsn_rgb_reference_bn_inception.caffemodel' ,  '--num_worker', '1', '--save_scores', 'myscores_fre.txt']

  parser = argparse.ArgumentParser()
  parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
  parser.add_argument('split', type=int, choices=[1, 2, 3],
                    help='on which split to test the network')
  parser.add_argument('modality', type=str, choices=['rgb', 'flow'])
  parser.add_argument('frame_path', type=str, help="root directory holding the frames")
  parser.add_argument('net_proto', type=str)
  parser.add_argument('net_weights', type=str)
  parser.add_argument('--rgb_prefix', type=str, help="prefix of RGB frames", default='img_')
  parser.add_argument('--flow_x_prefix', type=str, help="prefix of x direction flow images", default='flow_x_')
  parser.add_argument('--flow_y_prefix', type=str, help="prefix of y direction flow images", default='flow_y_')
  parser.add_argument('--num_frame_per_video', type=int, default=25,
                    help="prefix of y direction flow images")
  parser.add_argument('--save_scores', type=str, default=None, help='the filename to save the scores in')
  parser.add_argument('--num_worker', type=int, default=1)
  parser.add_argument("--caffe_path", type=str, default='./lib/caffe-action/', help='path to the caffe toolbox')
  parser.add_argument("--gpus", type=int, nargs='+', default=None, help='specify list of gpu to use')
  args = parser.parse_args()
  print(args)

  sys.path.append(os.path.join(args.caffe_path, 'python'))
  from pyActionRecog import parse_directory
  from pyActionRecog import parse_split_file
  from pyActionRecog.utils.video_funcs import default_aggregation_func
  from pyActionRecog.action_caffe import CaffeNet

# build neccessary information
  print(args.dataset)
  split_tp = parse_split_file(args.dataset)
  f_info = parse_directory(args.frame_path,
                         args.rgb_prefix, args.flow_x_prefix, args.flow_y_prefix)

  gpu_list = args.gpus

  eval_video_list = split_tp[args.split - 1][1]

  score_name = 'fc-action'

  if 1:
    eval_video_list =  [('ua',1)]
    print(eval_video_list[0])
    print(f_info)

  #if args.num_worker > 1:
#    pool = multiprocessing.Pool(args.num_worker, initializer=build_net)#
    #video_scores_rgb = pool.map(eval_video, eval_video_list)
  #else:
#    build_net()
#    video_scores_rgb = map(eval_video, eval_video_list)

 # video_pred = [np.argmax(default_aggregation_func(x[0])) for x in video_scores_rgb]
  #print(video_pred)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
