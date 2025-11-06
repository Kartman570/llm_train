from huggingface_hub import snapshot_download
import os

model_id = "mistralai/Mistral-7B-v0.1"
local_dir = os.path.expanduser("~/models/mistral-7b")

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    revision="main"
)

print(f"Model downloaded to {local_dir}")
