#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Твои пути / модель
BASE = "Qwen/Qwen2.5-1.5B"
ADAPTERS = "outputs/qwen25_1_5b_run/lora_adapters"
CACHE_DIR = "./hf_cache"

# ЖЁСТКО работаем на CPU, чтобы исключить сюрпризы MPS
device = torch.device("cpu")
dtype = torch.float32

print("[info] device:", device)

# === Токенайзер ===
tok = AutoTokenizer.from_pretrained(BASE, cache_dir=CACHE_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
eos_id = tok.eos_token_id

# === Базовая модель ===
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    cache_dir=CACHE_DIR,
    torch_dtype=dtype,
)
base.to(device)

# === LoRA-адаптеры ===
model = PeftModel.from_pretrained(base, ADAPTERS)
model.to(device)
model.eval()

if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()
if hasattr(model.config, "use_cache"):
    model.config.use_cache = True

print("[info] model device:", next(model.parameters()).device)
print("[info] model dtype :", next(model.parameters()).dtype)

# ---- Настройки генерации ----
prompt = "Here is some information about GiganticCorp:\n"
max_new_tokens = 64
temperature = 0.8
top_p = 0.9

enc = tok(prompt, return_tensors="pt")
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)


def top_p_sample_from_logits(logits, temperature: float, top_p: float):
    """
    logits: (1, vocab)
    возвращает next_token: (1, 1)
    """
    # масштабируем температурой
    logits = logits / max(temperature, 1e-6)

    # чистим NaN / inf
    logits = torch.nan_to_num(
        logits,
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    )

    # softmax -> вероятности
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    # нормализуем на всякий случай
    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = probs / torch.clamp(probs_sum, min=1e-12)

    # nucleus (top-p) фильтрация
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    sorted_probs[mask] = 0.0
    # нормализуем после усечения
    sorted_probs_sum = sorted_probs.sum(dim=-1, keepdim=True)
    sorted_probs = sorted_probs / torch.clamp(sorted_probs_sum, min=1e-12)

    # семплируем из обрезанного распределения
    next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_token_sorted)
    return next_token  # (1, 1)


print("[info] starting manual sampling generation...")
t0 = time.time()

with torch.no_grad():
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits[:, -1, :]  # (1, vocab)

        # Можно добавить диагностический вывод по NaN/inf, если интересно:
        # print(torch.isnan(logits).any(), torch.isinf(logits).any())

        next_token = top_p_sample_from_logits(
            logits,
            temperature=temperature,
            top_p=top_p,
        )  # (1, 1)

        token_id = next_token.item()
        if token_id == eos_id:
            break

        # добавляем токен в последовательность
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, device=device)],
            dim=-1,
        )

dt = time.time() - t0
print(f"[info] generation time: {dt:.2f} sec")
print("-----")
print(tok.decode(input_ids[0], skip_special_tokens=True))
