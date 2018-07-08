#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "Specify container name"
    exit 1
fi

if [[ $(docker ps -f name=$1 | tail -1 | grep $1) ]]
then
    docker exec -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -it $1 bash
elif [[ $(docker ps -f name=$1 -a | tail -1 | grep $1) ]]; then
    docker start $1
    docker exec -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -it $1 bash
else

    if [ -z "$2" ]
    then
        docker run -it -d -P -v $(readlink -e ../):/home/anon/crt-task \
            --name=$1 \
            --runtime=nvidia \
            -e COLUMNS="`tput cols`" \
            -e LINES="`tput lines`" \
            crt-task:env bash
    else
        docker run -it -d -P -v $(readlink -e ../):/home/anon/crt-task \
            --name=$1 \
            --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$2 \
            -e COLUMNS="`tput cols`" \
            -e LINES="`tput lines`" \
            crt-task:env bash
    fi

    docker exec -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -it $1 bash
fi
