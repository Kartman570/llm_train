#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model


# Чуть-чуть прибираемся в окружении
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser("LoRA fine-tune on MPS (Mac)")

    p.add_argument("--model", default="Qwen/Qwen2-7B",
                   help="HF model ID или локальный путь к базовой модели")
    p.add_argument("--data_path", required=True,
                   help="Путь к JSONL с {'text': '...'}")
    p.add_argument("--out_dir", default="./outputs/my_run",
                   help="Куда сохранить результаты обучения")
    p.add_argument("--cache_dir", default=None,
                   help="Куда кэшировать модель HF")

    # Тренировочные гиперпараметры (консервативные по умолчанию)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_seq_len", type=int, default=256,
                   help="Максимальная длина последовательности (токены)")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--gradient_accumulation", type=int, default=1)

    # Параметры LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Хочешь сразу получить слитую модель — добавь флаг
    p.add_argument("--merge_lora", action="store_true",
                   help="Слить адаптеры в базовую модель и сохранить как merged_model")

    return p.parse_args()


def ensure_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path: str) -> Dataset:
    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" not in obj or not isinstance(obj["text"], str):
                raise ValueError("Каждая строка JSONL должна содержать строковое поле 'text'")
            records.append({"text": obj["text"]})
    if not records:
        raise ValueError("Файл JSONL пуст или без валидных записей")
    return Dataset.from_list(records)


def main():
    args = parse_args()
    device = ensure_device()
    print(f"[info] device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # === Загрузка токенайзера и модели ===
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if device.type != "cpu" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    )

    # Для тренировки включаем gradient checkpointing и выключаем кеш
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    # === LoRA-конфиг ===
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_cfg)
    model.to(device)

    # === Данные ===
    ds = read_jsonl(args.data_path)

    def tokenize_fn(ex):
        tokens = tokenizer(
            ex["text"],
            truncation=True,
            max_length=args.max_seq_len,
            # padding НЕ делаем здесь, DataCollator паддит динамически по батчу
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = ds.map(tokenize_fn, remove_columns=["text"])
    train_ds = ds

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # === TrainingArguments без evaluation_strategy ===
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=(device.type != "cpu"),
        bf16=False,
        report_to=[],                # без W&B
        dataloader_pin_memory=False,
        gradient_checkpointing=True, # совместимо с 4.57.1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("[info] starting training...")
    trainer.train()
    print("[info] training finished")

    # === Сохраняем LoRA-адаптеры ===
    adapters_dir = os.path.join(args.out_dir, "lora_adapters")
    print(f"[info] saving LoRA adapters to {adapters_dir}")
    model.save_pretrained(adapters_dir)
    tokenizer.save_pretrained(args.out_dir)

    # === Опционально: merge в полную модель ===
    if args.merge_lora:
        print("[info] merging LoRA into base model...")
        # merge_and_unload вернёт базовую модель с применёнными дельтами
        merged = model.merge_and_unload()
        merged_dir = os.path.join(args.out_dir, "merged_model")
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[info] merged model saved to {merged_dir}")

    print("[done] all artifacts saved in:", args.out_dir)


if __name__ == "__main__":
    main()
