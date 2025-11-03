import argparse
from os.path import join

import pandas as pd
from datasets import Dataset
from loguru import logger
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
import torch
from peft import LoraConfig, get_peft_model, TaskType
from swanlab.integration.transformers import SwanLabCallback
import bitsandbytes as bnb
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import json


# 配置参数
def configuration_parameter():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for deepseek model")

    # 模型路径相关参数
    parser.add_argument("--model_name_or_path", type=str, default="./qwen_4b",
                        help="Path to the model directory downloaded locally")
    parser.add_argument("--output_dir", type=str,
                        default="",
                        help="Directory to save the fine-tuned model and checkpoints")

    # 数据集路径
    parser.add_argument("--train_file", type=str, default="./data/cold_start_datas.jsonl",
                        help="Path to the training data file in JSONL format")

    # 训练超参数
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the input")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Number of steps between logging metrics")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Number of steps between saving checkpoints")
    parser.add_argument("--save_total_limit", type=int, default=1,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup",
                        help="Type of learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps for learning rate scheduler")

    # LoRA 特定参数
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="Rank of LoRA matrices")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout rate for LoRA")

    # 分布式训练参数
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", -1)),
                        help="Local rank for distributed training")
    parser.add_argument("--distributed", type=bool, default=True, help="Enable distributed training")

    # 额外优化和硬件相关参数
    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--optim", type=str, default="adamw_torch",
                        help="Optimizer to use during training")
    parser.add_argument("--train_mode", type=str, default="lora",
                        help="lora or qlora")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--fp16", type=bool, default=True,
                        help="Use mixed precision (FP16) training")
    parser.add_argument("--report_to", type=str, default=None,
                        help="Reporting tool for logging (e.g., tensorboard)")
    parser.add_argument("--dataloader_num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--save_strategy", type=str, default="steps",
                        help="Strategy for saving checkpoints ('steps', 'epoch')")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay for the optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--remove_unused_columns", type=bool, default=True,
                        help="Remove unused columns from the dataset")

    args = parser.parse_args()
    return args


def find_all_linear_names(model, train_mode, target_module_filter: list[str] = None):
    assert train_mode in ['lora', 'qlora']
    # target_module_filter in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            if target_module_filter:
                if not any(t in name for t in target_module_filter):
                    continue
            names = name.split('.')
            lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    lora_module_names = list(lora_module_names)
    logger.info(f"LoRA target module names: {lora_module_names}")
    return lora_module_names

def add_new_special_token(tokenizer, model, new_tokens: list[str], policy: str = "mean"):
    """
    add new tokens to the tokenizer and model embedding, and expand the embedding layer.

    args:
        tokenizer: A transformers.AutoTokenizer instance
        model: A transformers.AutoModelForCausalLM instance
        new_tokens: A list of special tokens to be added, such as ["<question_type>", "</question_type>"]
    """
    assert isinstance(new_tokens, list), "new_tokens must be list[str]"
    print("before add new token:")
    print(f"Model input embedding size: {model.get_input_embeddings().weight.shape[0]}")
    print(f"Model output head size: {model.get_output_embeddings().weight.shape[0]}")
    # transformers.tokenizer.add_tokens() return the number of tokens successfully added.
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=False) # special_tokens=False-->当作普通token处理: 普通token,参与正常分词、训练、embedding. if special_tokens=True 代表当作特殊token处理 不参与分词 且特殊token的embedding会被特殊处理 比如<pad>、<bos>、<eos>、<unk>
    if num_added > 0:
        print(f"new {num_added} token: {new_tokens}")
        model.resize_token_embeddings(len(tokenizer)) # change embedding layer size and lm head layer size. that is, change the vocab size. 当调用model.resize_token_embeddings()时-->新增token的向量默认是随机初始化的-->这个随机向量与模型中其他经过数十亿token预训练、具有丰富语义信息的向量相比, 完全是“噪声” --> 模型在处理它时会感到非常困惑 --> 同样，在模型的lm head layer也增加了一个新的、随机初始化的logit-->模型在预测这个新token时完全是瞎猜。在SFT初期，当模型遇到这些新token时，会产生巨大的loss和梯度，这些梯度会反向传播并剧烈地更新模型参数，可能会破坏掉模型在预训练阶段学到的通用知识，导致"灾难性遗忘"
        print("after add new token")
        print(f"Model input embedding size: {model.get_input_embeddings().weight.shape[0]}")
        print(f"Model output head size: {model.get_output_embeddings().weight.shape[0]}")

        # Policy A
        # to avoid the gradient anomalies, need to init a embedding of new token
        if policy == "mean":
            with torch.no_grad():
                input_emb = model.get_input_embeddings().weight
                output_emb = model.get_output_embeddings().weight if model.get_output_embeddings() is not None else None

                # initialize the vector of the new token with the mean of the existing embeddings.
                mean_input_emb = input_emb[:-num_added].mean(dim=0, keepdim=True)
                input_emb[-num_added:] = mean_input_emb
                if output_emb is not None:
                    mean_output_emb = output_emb[:-num_added].mean(dim=0, keepdim=True)
                    output_emb[-num_added:] = mean_output_emb
            print("New special token embeddings have been initialized with the average of old embeddings.")
        elif policy == "other":
            raise NotImplementedError
    else:
        print("not new token added")

def setup_distributed(args):
    """初始化分布式环境"""
    if args.distributed:
        if args.local_rank == -1:
            raise ValueError("未正确初始化 local_rank，请确保通过分布式启动脚本传递参数，例如 torchrun。")

        # 初始化分布式进程组
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        print(f"分布式训练已启用，Local rank: {args.local_rank}")
    else:
        print("未启用分布式训练，单线程模式。")


# 加载模型
def load_model(args, train_dataset, data_collator):
    # 初始化分布式环境
    setup_distributed(args)
    # 自动分配设备
    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if args.fp16 else torch.bfloat16,
        "use_cache": False if args.gradient_checkpointing else True,
        "device_map": "auto" if not args.distributed else None,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    # 用于确保模型的词嵌入层参与训练
    model.enable_input_require_grads()
    # 将模型移动到正确设备
    if args.distributed:
        model.to(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # 哪些模块需要注入Lora参数
    target_modules = find_all_linear_names(model.module if isinstance(model, DDP) else model, args.train_mode)
    # lora参数设置
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False

    )
    use_bfloat16 = torch.cuda.is_bf16_supported()  # 检查设备是否支持 bf16
    # 配置训练参数
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        save_on_each_node=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to,
        seed=args.seed,
        optim=args.optim,
        local_rank=args.local_rank if args.distributed else -1,
        ddp_find_unused_parameters=False,  # 分布式参数检查优化
        fp16=args.fp16,
        bf16=not args.fp16 and use_bfloat16,
        remove_unused_columns=False
    )
    # 应用 PEFT 配置到模型
    model = get_peft_model(model.module if isinstance(model, DDP) else model, config)  # 确保传递的是原始模型
    print("model:", model)
    model.print_trainable_parameters()

    ### 展示平台
    swanlab_config = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "dataset": "single-data-3w"

    }
    swanlab_callback = SwanLabCallback(
        project="deepseek-finetune",
        experiment_name="deepseek-llm-7b-chat-lora",
        description="DeepSeek有很多模型，V2太大了，这里选择llm-7b-chat的，希望能让回答更加人性化",
        workspace=None,
        config=swanlab_config,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
    )
    return trainer


def process_data(data: dict, tokenizer, max_seq_length: int):
    conversations = data["conversations"]
    system_prompt = ""
    dialog_text = ""
    if len(conversations) > 0 and conversations[0]["role"] == "system":
        system_prompt = conversations[0]["content"].strip()
        conversations = conversations[1:]  # 去掉system这一条 避免多轮对话数据构造的时候 重复多次出现

    dialog_text = ""
    for conv in conversations:
        role = conv["role"]
        content = conv["content"].strip()

        if role == "user":
            dialog_text += f"<|im_start|>user\n{content}\n<|im_end|>\n"
        elif role == "assistant":
            dialog_text += f"<|im_start|>assistant\n{content}\n<|im_end|>\n"
        elif role == "system":
            dialog_text += f"<|im_start|>system\n{content}\n<|im_end|>\n"
    user_turn = ""
    assistant_turn = ""
    for conv in reversed(conversations):
        if conv["role"] == "assistant" and assistant_turn == "":
            assistant_turn = conv["content"].strip()
        elif conv["role"] == "user" and assistant_turn != "":
            user_turn = conv["content"].strip()
            break

    if not user_turn or not assistant_turn:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    input_text = ""
    if system_prompt:
        input_text += f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
    input_text += f"<|im_start|>user\n{user_turn}\n<|im_end|>\n<|im_start|>assistant\n"

    input_enc = tokenizer(
        input_text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    output_enc = tokenizer(
        assistant_turn + "<|im_end|>",
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )

    input_ids = (
        input_enc["input_ids"] + output_enc["input_ids"] + [tokenizer.eos_token_id]
    )
    attention_mask = (
        input_enc["attention_mask"] + output_enc["attention_mask"] + [1]
    )
    labels = (
        [-100] * len(input_enc["input_ids"]) + output_enc["input_ids"] + [tokenizer.eos_token_id]
    )

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# 训练部分
def main():
    args = configuration_parameter()
    print("*****************加载分词器*************************")
    # 加载分词器
    model_path = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    print("*****************处理数据*************************")
    # 处理数据
    # 获得数据
    data = pd.read_json(args.train_file, lines=True)
    train_ds = Dataset.from_pandas(data)
    train_dataset = train_ds.map(process_data,
                                 fn_kwargs={"tokenizer": tokenizer, "max_seq_length": args.max_seq_length},
                                 remove_columns=train_ds.column_names)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")
    print(train_dataset, data_collator)
    # 加载模型
    print("*****************训练*************************")
    trainer = load_model(args, train_dataset, data_collator)
    trainer.train()
    # 训练
    final_save_path = join(args.output_dir)
    trainer.save_model(final_save_path)


if __name__ == "__main__":
    main()
