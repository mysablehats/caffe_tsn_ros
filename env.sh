#!/usr/bin/env sh
ROSMASTERNAME = $1
shift
export ROS_MASTER_URI=http://$ROSMASTERNAME:11311
export ROS_IP=`hostname -I`
export PYTHONPATH=/temporal-segment-networks/caffe-action/python/caffe:/temporal-segment-networks/:$PYTHONPATH

exec /temporal-segment-networks/catkin_ws/devel/env.sh "$@"
