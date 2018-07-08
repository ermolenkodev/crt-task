#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "Specify environment"
    exit 1
fi

cp ../requirements.txt env

docker build -t crt-task:$1 $1 --build-arg UID=$(id -u $(whoami)) --build-arg GID=$(id -g $(whoami))

rm env/requirements.txt
