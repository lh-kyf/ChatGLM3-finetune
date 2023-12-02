#! /usr/bin/env bash

set -ex

RUN_NAME=ChatGLM3-Lora-sc1
BASE_MODEL_PATH=/data/zym_proj/huan/models/chatglm3-6b/snapshots/d3fe58f8a2c50bab217780ba8bd3ff76833d2d0c
DATASET_PATH=/data/zym_proj/huan/data/self_cognition_lh.json
LORA_PATH=/data/zym_proj/huan/projects/ChatGLM3-sft-train/configs/lora_config.json
OUTPUT_DIR=output/${RUN_NAME}

mkdir -p $OUTPUT_DIR

python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node 1 finetune_ChatGLM3.py \
--do_train \
--data_path $DATASET_PATH \
--output_dir $OUTPUT_DIR  \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--model_name_or_path $BASE_MODEL_PATH \
--lora_config $LORA_PATH \
--weight_decay 0.1 \
--learning_rate 2e-5 \
--model_max_length 1024 \
--bf16 True \
--gradient_accumulation_steps 32 \
--dataloader_num_workers 0 \
--peft_lora True \
# --remove_unused_columns=True\

# --deepspeed ./configs/dp_config_zero2.json \
# --use_flash_attention2 True

# --freeze_layers "embed_tokens" \
# --deepspeed ./dp_config_zero2.json 
# --fsdp "shard_grad_op"
# --deepspeed ./dp_config_zero2.json 
# --gradient_accumulation_steps 8
# --fsdp "full_shard offload"
# --peft_lora True \
# --deepspeed ./dp_config.json 
# --bf16 True 
# --gradient_accumulation_steps 64 
# --fsdp "shard_grad_op"
# --save_steps 100 
# --deepspeed ./dp_config.json \
# --use_cpu True \
# --bf16 True \
# --no_cuda True