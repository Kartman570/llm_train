WORK IN PROGRESS


# setup preparation

python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip

# PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# HuggingFace stack
pip3 install transformers datasets accelerate




# model download
to prevent ram/cache loading of models on script start, download them locally.
python3 model_downloader.py
that will save them in "~/models/"

later on you can address them as
```python
model = AutoModelForCausalLM.from_pretrained("~/models/mistral-7b")
tokenizer = AutoTokenizer.from_pretrained("~/models/mistral-7b")
```