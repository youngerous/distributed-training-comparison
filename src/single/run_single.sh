#!/bin/sh

EPOCH=50
BATCH_SIZE=128

LR=0.1
LR_DECAY_STEP_SIZE=25
LR_DECAY_GAMMA=0.1
WEIGHT_DECAY=0.0001

SEED=42

python src/single/main.py\
        --seed=${SEED}\
        --epoch=${EPOCH}\
        --batch-size=${BATCH_SIZE}\
        --lr=${LR}\
        --weight-decay=${WEIGHT_DECAY}\
        --lr-decay-step-size=${LR_DECAY_STEP_SIZE}\
        --lr-decay-gamma=${LR_DECAY_GAMMA}\
        --amp\
        --contain-test
