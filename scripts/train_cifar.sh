#!/usr/bin/env bash

export learningRate=0.1
export epoch_step="{60,120,160}"
export max_epoch=200
export learningRateDecay=0
export learningRateDecayRatio=0.2
export nesterov=true
export randomcrop_type=reflection
#export weightDecay=0.001

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
now=$(date +"%Y%m%d_%H_%M")
postfix=$1
# export save=logs/${model}_${RANDOM}${RANDOM}
export save=logs/${model}_${now}_${postfix}
mkdir -p $save
cp train.lua $0 models/${model}.lua ${save}/
th train.lua | tee $save/log.txt
