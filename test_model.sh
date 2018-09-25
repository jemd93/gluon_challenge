#!/usr/bin/env bash

PARAM_FILE=./model_params/$1
INPUT_DATA_FILE=$2
MODEL_NAME=$3
INPUT_STRING=$4 # Write string in double quotes. Ex : "Hello I'd like to "

MODE=$(sed -n 1p $PARAM_FILE | awk '{print $2}')
EMBED_SIZE=$(sed -n 2p $PARAM_FILE | awk '{print $2}')
HIDDEN_LAYERS=$(sed -n 3p $PARAM_FILE | awk '{print $2}')
HIDDEN_UNITS=$(sed -n 4p $PARAM_FILE | awk '{print $2}')
SEQ_LENGTH=$(sed -n 8p $PARAM_FILE | awk '{print $2}')
DROPOUT=$(sed -n 9p $PARAM_FILE | awk '{print $2}')

python test_gluon_model.py $INPUT_DATA_FILE $MODEL_NAME "${INPUT_STRING:0:${#INPUT_STRING}-1}" \
    --mode=$MODE --embed-size=$EMBED_SIZE --hidden-layers=$HIDDEN_LAYERS --hidden-units=$HIDDEN_UNITS \
    --seq-length=$SEQ_LENGTH --dropout=$DROPOUT