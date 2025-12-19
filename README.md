# PRODIGY_GA_01
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    
from datasets import load_dataset

dataset = load_dataset("text", data_files={"train": "data.txt"})
tokenized_dataset = dataset.map(tokenize_function, batched=True)
from transformers import Trainer, TrainingArguments
#Text Generation After Fine-Tuning
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
prompt = "Artificial intelligence in healthcare"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    inputs["input_ids"],
    max_length=150,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

