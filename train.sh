#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

save_dir=./result/$1   

save_code=$save_dir/code
if [ ! -d $save_code  ];then
  mkdir -p $save_dir
  mkdir -p $save_code
  echo mkdir $save_code
else
  echo dir exist
fi

cp ./*.py $save_code
cp ./*.txt $save_code

nohup python -u ./train.py $save_dir $1 1> $save_dir/A_log.txt 2>&1 &
