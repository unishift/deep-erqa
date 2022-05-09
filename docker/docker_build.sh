#!/bin/bash

ENV_NAME=22e_lya
CONTAINER_NAME=${ENV_NAME}-research

nvidia-docker build \
       --network=host \
       --tag=${CONTAINER_NAME}\
       --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) \
       .
