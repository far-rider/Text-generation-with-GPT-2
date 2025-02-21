from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Set and load the  path to your dataset file
file_path = r"C:\Users\user\gpt\data.txt"
dataset = load_dataset("text", data_files={"train": file_path})

split_dataset = dataset["train"].train_test_split(test_size=0.1)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


# Load tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token  

#tokenization
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["text"],
        truncation=True,  
        padding="max_length",
        max_length=512
    )
    
    
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    
    return tokenized_output

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)


#  Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch", 
    save_strategy="epoch",
    num_train_epochs=3,
    logging_dir="./logs"
)


# train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval 
)
 

trainer.train()


model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")





# Test the fine-tuned model and save it
from transformers import pipeline

generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer="./gpt2-finetuned")
prompt = "i am a legend"
result = generator(prompt, max_length=100, num_return_sequences=1)
print(result[0]['generated_text'])