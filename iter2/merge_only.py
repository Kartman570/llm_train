#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    p = argparse.ArgumentParser("Merge LoRA adapters into a base model (no training).")
    p.add_argument("--base", required=True, help="HF model id или локальный путь к БАЗОВОЙ модели (та же, на которой обучались адаптеры)")
    p.add_argument("--adapters", required=True, help="Путь к каталогу LoRA адаптеров (например outputs/my_run/lora_adapters)")
    p.add_argument("--out_dir", required=True, help="Куда сохранить слитую модель (новая папка, НЕ оригинальная база)")
    p.add_argument("--cache_dir", default=None, help="Необязательно: локальный кэш HF")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # dtype и устройство (для Mac — mps, иначе cpu)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    # грузим базу
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    ).to(device)

    # грузим адаптеры поверх базы и сливаем
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(base, args.adapters)
    merged = peft_model.merge_and_unload()  # вернёт базовую модель c применёнными весами LoRA

    # сохраняем слитую модель + токенайзер
    merged.save_pretrained(args.out_dir)
    tok = AutoTokenizer.from_pretrained(args.base, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(args.out_dir)

    print(f"[done] merged model saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
