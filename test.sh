#!/bin/sh

EXT="$1"

if [ "$EXT" = "" ]; then
    echo usage: $0 log_ext
    exit
fi

echo Using extension: $EXT

[ ! -f .env ] || export $(grep -v '^#' .env | xargs)

LLM_MODEL_TYPE=huggingface

HUGGINGFACE_MODEL_NAME_OR_PATH="lmsys/fastchat-t5-3b-v1.0"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/fastchat-t5-3b-v1.0_${EXT}.log


HUGGINGFACE_MODEL_NAME_OR_PATH="TheBloke/wizardLM-7B-HF"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/wizardLM-7B-HF_${EXT}.log


HUGGINGFACE_MODEL_NAME_OR_PATH="TheBloke/vicuna-7B-1.1-HF"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/vicuna-7B-1.1-HF_${EXT}.log


HUGGINGFACE_MODEL_NAME_OR_PATH="nomic-ai/gpt4all-j"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/gpt4all-j_${EXT}.log


# HUGGINGFACE_MODEL_NAME_OR_PATH="nomic-ai/gpt4all-falcon"
# echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
# python test.py 2>&1 | tee ./data/logs/gpt4all-falcon_${EXT}.log

LLM_MODEL_TYPE=stablelm

STABLELM_MODEL_NAME_OR_PATH="stabilityai/stablelm-tuned-alpha-7b"
echo Testing $STABLELM_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/stablelm-tuned-alpha-7b_${EXT}.log


STABLELM_MODEL_NAME_OR_PATH="OpenAssistant/stablelm-7b-sft-v7-epoch-3"
echo Testing $STABLELM_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/stablelm-7b-sft-v7-epoch-3_${EXT}.log


LLM_MODEL_TYPE=mosaicml
MOSAICML_MODEL_NAME_OR_PATH="nomic-ai/gpt4all-mpt"
echo Testing $MOSAICML_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/gpt4all-mpt_${EXT}.log


LLM_MODEL_TYPE=huggingface
HUGGINGFACE_MODEL_NAME_OR_PATH="HuggingFaceH4/starchat-beta"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
LOAD_QUANTIZED_MODEL=8bit python test.py 2>&1 | tee ./data/logs/starchat-beta_${EXT}.log


HUGGINGFACE_MODEL_NAME_OR_PATH="../../models/starcoder"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
LOAD_QUANTIZED_MODEL=8bit python test.py 2>&1 | tee ./data/logs/starcoder_${EXT}.log
