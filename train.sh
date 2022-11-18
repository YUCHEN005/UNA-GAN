#!/usr/bin/env bash

cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node01"


export CUDA_VISIBLE_DEVICES=0,1

source activate audio

#python -m visdom.server

$cmd log/xx.log \
python train.py --dataroot ./datasets/xx --name xx --CUT_mode CUT --checkpoints_dir checkpoints_xx
