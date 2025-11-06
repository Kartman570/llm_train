#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# --- Без лишнего шума и онлайн-логгеров (см. совет в статье) ---
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune on Apple Silicon (MPS)")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B", help="HF model id or local path")
    p.add_argument("--data_path", required=True, help="Path to JSONL with {'text': ...}")
    p.add_argument("--out_dir", default="./outputs/lora_run", help="Output dir")
    p.add_argument("--cache_dir", default=None, help="Optional local HF cache for model")
    p.add_argument("--merge_lora", action="store_true", help="Merge LoRA into base at the end")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--gradient_accumulation", type=int, default=4,
                   help="Increase effective batch size: tokens * GA")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def ensure_mps():
    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    # Фоллбек на CPU, если MPS недоступен
    print("Using CPU")
    return torch.device("cpu")


def read_jsonl(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text" not in obj or not isinstance(obj["text"], str):
                raise ValueError("Each JSONL line must contain a string field 'text'")
            records.append({"text": obj["text"]})
    if not records:
        raise ValueError("No records found in JSONL")
    return Dataset.from_list(records)


def main():
    args = parse_args()
    device = ensure_mps()
    print(f"[info] Using device: {device}")

    # Загружаем токенайзер и модель. Никакого bitsandbytes/4bit — MPS не поддерживает.
    # (см. статью и issues bitsandbytes)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        # безопасная установка pad к eos
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16  # на MPS обычно это OK и экономит память
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    )

    # Включаем gradient checkpointing для экономии памяти
    model.gradient_checkpointing_enable()

    # Настройки LoRA: целевые линейные слои под «ламоподобные/трансформерные» модели.
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Подготавливаем модель к LoRA
    model = get_peft_model(model, lora_cfg)

    # Переносим на MPS/CPU
    model.to(device)

    # Данные
    ds = read_jsonl(args.data_path)

    def tokenize_fn(ex):
        # Простое SFT «по тексту» (как отмечено в обсуждениях MLX/LoRA примеров)
        # обрезаем/паддим до max_seq_len
        tokens = tokenizer(
            ex["text"],
            max_length=args.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        # labels = input_ids (стандартный LM objective)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    ds = ds.map(tokenize_fn, remove_columns=["text"])
    # Небольшой dev-сплит не делаем (можно добавить при желании)
    train_ds = ds

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    os.makedirs(args.out_dir, exist_ok=True)

    # Токенов в батче ≈ max_seq_len * (batch_size=1) * grad_accumulation
    # На 16 ГБ это безопасный минимум.
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
        report_to=[],  # без W&B
        dataloader_pin_memory=False,  # на MPS часто лучше отключать
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Сохраняем только LoRA-адаптеры
    print("[info] Saving LoRA adapters...")
    model.save_pretrained(os.path.join(args.out_dir, "lora_adapters"))
    tokenizer.save_pretrained(args.out_dir)

    if args.merge_lora:
        # Сливаем адаптеры в базовую модель и сохраняем итог
        # На Mac это может занять время/память — делаем последним шагом.
        print("[info] Merging LoRA into base model...")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            cache_dir=args.cache_dir,
            torch_dtype=torch_dtype,
        ).to(device)
        peft_model = PeftModel(base, os.path.join(args.out_dir, "lora_adapters"))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(os.path.join(args.out_dir, "merged_model"))
        tokenizer.save_pretrained(os.path.join(args.out_dir, "merged_model"))
        print("[info] Merged model saved.")

    print("[done] Training complete. Artifacts in:", args.out_dir)
    print("[hint] Модель/веса кэшируются в ~/.cache/huggingface/hub (или в --cache_dir). "
          "Повторно скачивать не придётся.")


if __name__ == "__main__":
    main()
