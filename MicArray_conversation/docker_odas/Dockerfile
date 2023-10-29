# Copyright 2020 The Cartographer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM osrf/ros:melodic-desktop-full
#FROM ubuntu:22.04

#ARG CARTOGRAPHER_VERSION=master

# ROS Noetic's base image doesn't ship with sudo and git.
RUN apt-get update && apt-get install -y sudo git



RUN apt-get install -y vim net-tools 
RUN apt-get install -y cmake g++ pkg-config
ENV ROS_DISTRO="melodic"
#RUN apt-get install -y ros-$ROS_DISTRO-jackal*

#RUN apt-get install -y ros-$ROS_DISTRO-gazebo-* ros-$ROS_DISTRO-dwa-local-planner

#--------------------
COPY scripts/install_odas.sh /root/work/scripts/
COPY scripts/run_odas.sh /root/work/scripts/
RUN /root/work/scripts/install_odas.sh

#ADD /scripts /scripts
#RUN bash /scripts/install_scripts_for_docker.bash
#--------------------



#COPY scripts/ros_entrypoint.sh /

#ENTRYPOINT ["/ros_entrypoint.sh"]
#CMD ["/bin/bash"]

#RUN rm -rf /var/lib/apt/lists/*
# A BTRFS bug may prevent us from cleaning up these directories.
# https://btrfs.wiki.kernel.org/index.php/Problem_FAQ#I_cannot_delete_an_empty_directory
#RUN rm -rf cartographer_ros catkin_ws || true