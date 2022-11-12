#!/usr/bin/env bash

cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node01"


export CUDA_VISIBLE_DEVICES=0,1

source activate cutmodel

#python -m visdom.server

$cmd log/checkpoints_rats_10min_1h.log \
CUDA_VISIBLE_DEVICES=0,1 \
python train.py --dataroot ./datasets/rats_10min_1h --name rats_10min_1h --CUT_mode CUT --checkpoints_dir checkpoints_rats_10min_1h
