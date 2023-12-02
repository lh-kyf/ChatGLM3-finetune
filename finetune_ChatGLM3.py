import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from data_utils import jload
from PROMPT import PROMPT_DICT

import sys

print("当前Python解释器路径:", sys.executable)

import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer
from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100

logging.basicConfig(filename='train.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    peft_lora: bool = field(default=False, metadata={"help": "Whether to use PEFT."})
    lora_config: Optional[str] = field(default=None, metadata={"help": "Path to the PEFT config."})
    freeze_layers: Optional[str] = field(default=None, metadata={"help": "Layers to freeze."})
    use_flash_attention2: bool = field(default=False, metadata={"help": "Whether to use flash attention."})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=True)  # 确保移除不需要的列
    model_max_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            max_length=tokenizer.model_max_length,
            padding="longest",
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_path.split(",")
        print(data_path)
        list_data_dict = []
        for path in data_path:
            data = jload(path)
            list_data_dict.extend(data)
        print("Data total length: " + str(len(list_data_dict)))
        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    # def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    #     return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = dict(input_ids=self.input_ids[i], labels=self.labels[i])
        # print(f"Returning item from dataset at index {i}: {item}")
        return item


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        try:
            # print(f"Received instances in collator: {instances}")

            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
        except KeyError as e:
            print(f"KeyError encountered: {e}")
            print(f"The problematic instance: {instances}")
            raise




def freeze_model_layers(model, freeze_layer_name):
    print(freeze_layer_name)
    for name, param in model.model.named_parameters():
        if name in freeze_layer_name:
            param.requires_grad = False


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    print(train_dataset[0])
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_flash_attention2:
        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            use_flash_attention_2=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("You are using flash attention 2!\n")
    else:
        model = transformers.AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side='left',
        # padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )

    if model_args.freeze_layers:
        freeze_model_layers(model, model_args.freeze_layers.split(","))
    if model_args.peft_lora:
        if model_args.lora_config is None:
            raise ValueError("Please specify the path to the PEFT config.")
        lora_config = LoraConfig(**LoraConfig.from_json_file(model_args.lora_config))
        # model.add_adapter(lora_config, adapter_name="adapter_lora")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("You are using lora model!\n")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    print(f"self._signature_columns: {trainer._signature_columns}")  # 查看签名列
    trainer._signature_columns = ['input_ids', 'labels']  # 设置签名列
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
