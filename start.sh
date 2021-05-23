#!/bin/bash
cd "${0%/*}"
xhost +local:root

source ./arg.sh

if [ "$(uname)" == "Darwin" ]
then
    export HOST_IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
    xhost +$HOST_IP
    export DISPLAY=$HOST_IP":0"
fi

docker-compose down --remove-orphans
docker-compose -f docker-compose.yaml up -d --build

if [ ${DEV} == "yes" ]
then
    echo "${red}In DEV Environment";
else
    echo "${red}Not in Dev.";
fi
