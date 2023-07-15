#!/bin/sh

EXT="$1"

if [ "$EXT" = "" ]; then
    echo usage: $0 log_ext
    exit
fi

echo Using extension: $EXT

[ ! -f .env ] || export $(grep -v '^#' .env | xargs)

export LLM_MODEL_TYPE=openai
export OPENAI_MODEL_NAME="gpt-3.5-turbo"
echo Testing openai-${OPENAI_MODEL_NAME}
python test.py 2>&1 | tee ./data/logs/openai-${OPENAI_MODEL_NAME}_${EXT}.log

export OPENAI_MODEL_NAME="gpt-4"
echo Testing openai-${OPENAI_MODEL_NAME}
python test.py 2>&1 | tee ./data/logs/openai-${OPENAI_MODEL_NAME}_${EXT}.log

export LLM_MODEL_TYPE=huggingface

export HUGGINGFACE_MODEL_NAME_OR_PATH="lmsys/fastchat-t5-3b-v1.0"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/fastchat-t5-3b-v1.0_${EXT}.log


export HUGGINGFACE_MODEL_NAME_OR_PATH="TheBloke/wizardLM-7B-HF"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/wizardLM-7B-HF_${EXT}.log


export HUGGINGFACE_MODEL_NAME_OR_PATH="TheBloke/vicuna-7B-1.1-HF"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/vicuna-7B-1.1-HF_${EXT}.log


export HUGGINGFACE_MODEL_NAME_OR_PATH="nomic-ai/gpt4all-j"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/gpt4all-j_${EXT}.log


# export HUGGINGFACE_MODEL_NAME_OR_PATH="nomic-ai/gpt4all-falcon"
# echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
# python test.py 2>&1 | tee ./data/logs/gpt4all-falcon_${EXT}.log

export LLM_MODEL_TYPE=stablelm

# export STABLELM_MODEL_NAME_OR_PATH="stabilityai/stablelm-tuned-alpha-7b"
# echo Testing $STABLELM_MODEL_NAME_OR_PATH
# python test.py 2>&1 | tee ./data/logs/stablelm-tuned-alpha-7b_${EXT}.log


export STABLELM_MODEL_NAME_OR_PATH="OpenAssistant/stablelm-7b-sft-v7-epoch-3"
echo Testing $STABLELM_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/stablelm-7b-sft-v7-epoch-3_${EXT}.log


export LLM_MODEL_TYPE=mosaicml
export MOSAICML_MODEL_NAME_OR_PATH="mosaicml/mpt-7b-instruct"
echo Testing $MOSAICML_MODEL_NAME_OR_PATH
python test.py 2>&1 | tee ./data/logs/mpt-7b-instruct_${EXT}.log


# export MOSAICML_MODEL_NAME_OR_PATH="mosaicml/mpt-30b-instruct"
# echo Testing $MOSAICML_MODEL_NAME_OR_PATH
# LOAD_QUANTIZED_MODEL=4bit python test.py 2>&1 | tee ./data/logs/mpt-30b-instruct_${EXT}.log

export LLM_MODEL_TYPE=huggingface
export HUGGINGFACE_MODEL_NAME_OR_PATH="HuggingFaceH4/starchat-beta"
echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
LOAD_QUANTIZED_MODEL=8bit python test.py 2>&1 | tee ./data/logs/starchat-beta_${EXT}.log


# export HUGGINGFACE_MODEL_NAME_OR_PATH="../../models/starcoder"
# echo Testing $HUGGINGFACE_MODEL_NAME_OR_PATH
# LOAD_QUANTIZED_MODEL=8bit python test.py 2>&1 | tee ./data/logs/starcoder_${EXT}.log
