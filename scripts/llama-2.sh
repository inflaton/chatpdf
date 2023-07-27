#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

nvidia-smi

export TRANSFORMERS_CACHE=/common/scratch/users/d/dh.huang.2023/transformers

EXT=cluster_a40

export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-7b-chat-hf_${EXT}.log


export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-13b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-13b-chat-hf_${EXT}.log


export LOAD_QUANTIZED_MODEL=4bit
export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-70b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-70b-chat-hf_${EXT}.log

# export HUGGINGFACE_MODEL_NAME_OR_PATH="Panchovix/LLaMA-2-70B-GPTQ-transformers4.32.0.dev0"
# echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
# python test.py 2>&1 | tee ./data/logs/LLaMA-2-70B-GPTQ-transformers4.32.0.dev0_${EXT}.log
