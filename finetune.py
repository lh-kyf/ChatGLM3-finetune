import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import json
from peft import TaskType, get_peft_model, LoraConfig, PromptEncoderConfig, PromptEncoderReparameterizationType

MAX_LENGTH = 512


# 数据处理流程,参考GLM3仓库:https://github.com/THUDM/ChatGLM3/blob/main/finetune_chatmodel_demo/preprocess_utils
def process_func(example, tokenizer):
    MAX_LENGTH = 512
    input_ids, labels = [], []
    instruction = tokenizer.encode(text="\n".join(["", "你是一个诚实而优秀的中国人助手。", "",
                                    example["instruction"] + example["input"] + ""]).strip() + "\n",
                                    add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
    response = tokenizer.encode(text=example["output"], add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    input_ids = instruction + response + [tokenizer.eos_token_id]
    labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id]
    pad_len = MAX_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [tokenizer.pad_token_id] * pad_len
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

    return {
        "input_ids": input_ids,
        "labels": labels
    }


args = TrainingArguments(
    # output_dir="./output/ChatGLM3-pt-sc2",
    output_dir="./output/ChatGLM3-Lora-sc2",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    logging_steps=1,
    num_train_epochs=3,
    remove_unused_columns=True
)

if "__main__" == __name__:
    # 加载数据
    file_path = '/data/zym_proj/huan/data/self_cognition_lh.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 将JSON数据转换为Dataset对象
    dataset = Dataset.from_list(data)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/zym_proj/huan/models/chatglm3-6b/snapshots/d3fe58f8a2c50bab217780ba8bd3ff76833d2d0c", trust_remote_code=True)

    # 应用数据处理函数
    processed_dataset = dataset.map(lambda example: process_func(example, tokenizer))
    print("Dataset size:", len(processed_dataset))
    print(processed_dataset[0])
    print(processed_dataset[1])

    # 创建模型
    model = AutoModelForCausalLM.from_pretrained("/data/zym_proj/huan/models/chatglm3-6b/snapshots/d3fe58f8a2c50bab217780ba8bd3ff76833d2d0c", trust_remote_code=True, low_cpu_mem_usage=True)
    model.to("cuda")

    # 创建loRA参数
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules={"query_key_value"}, r=8, lora_alpha=32)

    # 创建P-tuning参数
    # config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10,
    #                             encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
    #                             encoder_dropout=0, encoder_num_layers=5, encoder_hidden_size=1024)

    # 添加adapter
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 指定GLM的Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    # 指定训练参数。
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()
