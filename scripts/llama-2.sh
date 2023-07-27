#!/bin/sh
cd ..
pwd

EXT=cluster

export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-7b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-7b-chat-hf_${EXT}.log


export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-13b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-13b-chat-hf_${EXT}.log


export HUGGINGFACE_MODEL_NAME_OR_PATH="meta-llama/Llama-2-70b-chat-hf"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/Llama-2-13b-chat-hf_${EXT}.log
