import random
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer

with open('reuters.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.shuffle(lines)

train_lines = lines[:int(0.8 * len(lines))]
valid_lines = lines[int(0.8 * len(lines)):]

with open('train.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open('valid.txt', 'w', encoding='utf-8') as f:
    f.writelines(valid_lines)

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="train.txt",
    block_size=128,
)

valid_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="valid.txt",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    num_train_epochs=0.5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    push_to_hub=False,
    gradient_accumulation_steps=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("finetuned_model")
tokenizer.save_pretrained("finetuned_tokenizer")
