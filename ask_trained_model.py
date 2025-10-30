from transformers import pipeline

pipe = pipeline("text-generation", model="./finetuned-heliotek", tokenizer="./finetuned-heliotek")

prompt = "What do you know about heliotech?"
print(pipe(prompt, max_new_tokens=50)[0]["generated_text"])
