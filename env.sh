#!/usr/bin/env sh

export ROS_MASTER_URI=http://SATELLITE-S50-B:11311
export ROS_IP=`hostname -I`
export PYTHONPATH=/temporal-segment-networks/caffe-action/python/caffe:/temporal-segment-networks/:$PYTHONPATH

exec /temporal-segment-networks/catkin_ws/devel/env.sh "$@"
