#!/bin/bash

ENV_NAME=22e_lya
CONTAINER_NAME=${ENV_NAME}-research

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

PATH_TO_EXPERIMENTS="/home/unishift/restoration-metric"
PATH_TO_DATA="/mnt/hdd/unishift"
HOST_NAME=$(hostname -s)

docker run -t -d --rm\
	   --runtime=nvidia\
	   --gpus '"device=1"'\
     --net=host\
     --shm-size=4g\
     --user=$(id -u):$(id -g)\
     --name=${CONTAINER_NAME}\
     -v $PATH_TO_EXPERIMENTS:/home/experiments\
     -v $PATH_TO_DATA:/home/data\
	   ${CONTAINER_NAME}\
	   zsh
