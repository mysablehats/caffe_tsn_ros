#!/bin/sh
source catkin_ws/devel/setup.bash
tmux new-session -d 'bash'
tmux split-window -h bash
tmux send -t 0:0.0 "source catkin_ws/devel/setup.bash" C-m
tmux send -t 0:0.1 "source catkin_ws/devel/setup.bash" C-m
tmux send -t 0:0.0 "roslaunch caffe_tsn_ros conv.launch img:=/videofiles/image_raw" C-m
tmux send -t 0:0.1 "rostopic hz /scores" C-m
#tmux new-window 'mutt'
tmux -2 attach-session -d

















#!/bin/sh
# #roslaunch caffe_tsn_ros cf3.launch
# #source catkin_ws/devel/setup.bash
# tmux new-session -s mysession -d 'ls'
# tmux splitw -v -d
# tmux attach -t mysession
