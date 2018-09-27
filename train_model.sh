#!/usr/bin/env bash

PARAM_FILE=./model_params/$1
INPUT_DATA_FILE=$2
OUTPUT_DATA_FILE=$3
MODEL_NAME=$4
INTENT=$5

MODE=$(sed -n 1p $PARAM_FILE | awk '{print $2}')
EMBED_SIZE=$(sed -n 2p $PARAM_FILE | awk '{print $2}')
HIDDEN_LAYERS=$(sed -n 3p $PARAM_FILE | awk '{print $2}')
HIDDEN_UNITS=$(sed -n 4p $PARAM_FILE | awk '{print $2}')
CLIP=$(sed -n 5p $PARAM_FILE | awk '{print $2}')
EPOCHS=$(sed -n 6p $PARAM_FILE | awk '{print $2}')
BATCH_SIZE=$(sed -n 7p $PARAM_FILE | awk '{print $2}')
SEQ_LENGTH=$(sed -n 8p $PARAM_FILE | awk '{print $2}')
DROPOUT=$(sed -n 9p $PARAM_FILE | awk '{print $2}')
OPTIMIZER=$(sed -n 10p $PARAM_FILE | awk '{print $2}')

python ./model_scripts/train_gluon_model.py $INPUT_DATA_FILE $OUTPUT_DATA_FILE $MODEL_NAME $INTENT \
    --mode=$MODE --embed-size=$EMBED_SIZE --hidden-layers=$HIDDEN_LAYERS --hidden-units=$HIDDEN_UNITS \
    --clip=$CLIP --epochs=$EPOCHS --batch-size=$BATCH_SIZE --seq-length=$SEQ_LENGTH \
    --dropout=$DROPOUT --optimizer=$OPTIMIZER