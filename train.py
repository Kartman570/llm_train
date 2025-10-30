from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


# PREPARE DATA
dataset = load_dataset("json", data_files="data/heliotek.jsonl")
dataset = dataset["train"]

path = os.path.expanduser("~/models/tinyllama")
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)

# === FIX: Add padding token if missing ===
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
# =========================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])



# CONFIGURE TRAINING

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./finetuned-heliotek",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,      # M1 memory limitation
    gradient_accumulation_steps=8,      # batch imitation
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)





# TRAIN MODEL WITH DATA
trainer.train()
trainer.save_model("./finetuned-heliotek")
tokenizer.save_pretrained("./finetuned-heliotek")
