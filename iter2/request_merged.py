# infer_merged.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MERGED_DIR = "outputs/my_run/merged_model"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MERGED_DIR)
if tok.pad_token is None: tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.float16 if device!="cpu" else torch.float32).to(device).eval()

prompt = "Сформулируй ответ на основе новых знаний:"
ids = tok(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))
