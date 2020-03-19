#! /bin/bash
clear

root_dir=$(cd `dirname $0`;pwd)
script_name="deduplicate.py"


function stop_process(){
    pids=$(ps -ef | grep $script_name | grep -v "$0" | grep -v "grep" | awk '{print $2}')
    echo "stopping process ... "
    for id in $pids
    do
        kill -9 $id
        echo "killed $id"
    done
    echo "finished "
}

function check_status(){
    pids=($(ps -ef | grep $script_name | grep -v "$0" | grep -v "grep" | awk '{print $2}'))
    echo ${#pids[*]} "processes are runing"
}

run_process="/home/dasen/anaconda3/bin/python3 $root_dir/$script_name"
function start_process(){
    stop_process
    echo " starting process ... "
    nohup ${run_process} >> ./log/nohup.out &
    sleep 1
    echo "finished "
}

function readin_promote(){
    read -p "what do you want? 0: quit, 1: start dedup, 2: stop dedup, 3: check status : " choice
    if [ $choice == 0 ];then
        return 2
    elif [ $choice == 1 ]; then
        start_process
    elif [ $choice == 2 ]; then
        stop_process
    elif [ $choice == 3 ]; then
        check_status
    else
        echo "invalid input"
    fi
    }


while :
do
    readin_promote
    if [ "$?" == 2 ];then
        break
    fi

done
