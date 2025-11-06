

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install 'torch' 'torchvision' -f https://download.pytorch.org/whl/metal.html


pip install transformers datasets peft accelerate safetensors









A) Только адаптеры + инференс с ними
1) Обучить и сохранить адаптеры
python train_lora_mps.py \
  --model Qwen/Qwen2.5-1.5B \
  --data_path data/train.jsonl \
  --out_dir outputs/qwen25_1_5b_run \
  --cache_dir ./hf_cache



LoRA адаптеры будут в: outputs/my_run/lora_adapters/

База кэшируется в ./hf_cache (или в ~/.cache/huggingface/hub)

2) Запрос к базе с учётом адаптеров
python infer_with_lora.py











B) Адаптеры + немедленный merge в новую модель
1) Обучить, сохранить адаптеры и сразу слить их в базу
python finetune_lora_mps.py \
  --data_path data/train.jsonl \
  --out_dir outputs/my_run \
  --cache_dir ./hf_cache \
  --merge_lora


Адаптеры: outputs/my_run/lora_adapters/

Слитая полная модель: outputs/my_run/merged_model/
(оригинальная база не перезаписывается)

2) Запрос к новой слитой модели (без PEFT)

Сохрани мини-скрипт как infer_merged.py:

# infer_merged.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MERGED_DIR = "outputs/my_run/merged_model"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MERGED_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MERGED_DIR,
    torch_dtype=torch.float16 if device!="cpu" else torch.float32
)
model.to(device)
model.eval()

prompt = "Сформулируй, основываясь на новых знаниях: "
inputs = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))


Запуск:

python infer_merged.py





MERGE ONLY (GOT ADAPTERS AND READY TO MERGE)
python merge_lora_only.py \
  --base Qwen/Qwen2-7B \
  --adapters outputs/my_run/lora_adapters \
  --out_dir outputs/my_run/merged_model \
  --cache_dir ./hf_cache
