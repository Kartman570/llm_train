from huggingface_hub import snapshot_download
import os

model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
local_dir = os.path.expanduser("~/models/tinyllama")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
