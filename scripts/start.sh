#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export LOAD_QUANTIZED_MODEL=4bit
export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-70b-chat-hf"
export TRANSFORMERS_CACHE=/common/scratch/users/d/dh.huang.2023/transformers

make start
