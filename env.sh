#!/usr/bin/env sh
export ROS_MASTER_URI=http://$1:11311
export ROS_IP=`hostname -I`
export PYTHONPATH=/temporal-segment-networks/caffe-action/python/caffe:/temporal-segment-networks/:$PYTHONPATH

#env | grep ROS
shift
exec /temporal-segment-networks/catkin_ws/devel/env.sh "$@"
