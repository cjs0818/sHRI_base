xhost +local:root

XAUTH=/tmp/.docker.xauth
#if [ ! -f $XAUTH ]
#then
#    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
#    if [ ! -z "$xauth_list" ]
#    then
#        echo $xauth_list | xauth -f $XAUTH nmerge -
#    else
#        touch $XAUTH
#    fi
#    chmod a+r $XAUTH
#fi

# MODIFY BELOW (NOTE(jwd) - you may need to change the network id `wlp5s0` below)
#export DOCKER_HOST_PC=$(ifconfig wlp5s0 | awk '/inet / {print $2}')  # for joesbox
#export DOCKER_HOST_PC=$(ifconfig wlp9s0 | awk '/inet / {print $2}')  # for Lenovo X1
#export DOCKER_HOST_PC=$(ifconfig wlxb0a7b9defe3d | awk '/inet / {print $2}')  # for Lenovo X1
#export DOCKER_HOST_PC=$(ifconfig enp0s31f6 | awk '/inet / {print $2}')  # for Office Ubuntu PC
#export DOCKER_HOST_PC=$(ifconfig enp0s5 | awk '/inet / {print $2}')  # for Office Ubuntu PC
export DOCKER_HOST_PC="localhost"
export ROS_MASTER_URI_PC=${DOCKER_HOST_PC}
#export ROS_MASTER_URI_PC=192.168.1.6
export ROS_PORT=11311
#export ROS_MASTER_CONTAINER=jackal_os1-rosmaster
export ROS_MASTER_CONTAINER=odas-rosmaster
export IMAGE_ID=odas
# END MODIFY

docker run -it --rm \
    --volume="${XAUTH}:${XAUTH}" \
    --volume /etc/localtime:/etc/localtime:ro \
    --volume ~/Downloads:/root/Downloads \
    --volume ${PWD}/Shared_folder:/root/Shared_folder \
    --volume="/dev:/dev" \
    --env="NVIDIA_DRIVER_CAPABILITIES=all" \
	--env="DISPLAY=${DISPLAY}" \
	--env="QT_X11_NO_MITSHM=1" \
	--env="XAUTHORITY=${XAUTH}" \
    --env "ROS_MASTER_URI=http://$ROS_MASTER_URI_PC:$ROS_PORT" \
    --env "ROS_HOSTNAME=$DOCKER_HOST_PC" \
    --name $ROS_MASTER_CONTAINER \
    --privileged \
    --network=host \
    $IMAGE_ID
    #roscore
    #/bin/bash

#    --platform linux/amd64 \
#    --runtime=nvidia \
#    -v ${PWD}:/catkin_ws/src \
#    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \