### **PRODIGY_GA_01**

### **Fine-Tuning GPT-2 with Transformers**

This project demonstrates how to fine-tune a **GPT-2** model using the **Transformers** library by Hugging Face. The dataset is loaded from a text file, tokenized, and then trained using the `Trainer` API.

## **Features**
- Load and preprocess text data  
- Fine-tune GPT-2 on a custom dataset  
- Save and reload the trained model  
- Generate text with the fine-tuned model  

## **📦 Installation**
First, install the required dependencies:  
```bash
pip install transformers datasets torch accelerate
```

If you're using a **GPU**, configure `accelerate` for faster training:  
```bash
accelerate config
```

---

## **📂 Dataset**
Place your dataset (`data.txt`) in the appropriate directory and update the file path in `file_path` inside the script.

---

## **📜 How to Run**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/gpt2-finetuning.git
   cd gpt2-finetuning
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

3. After training, test the fine-tuned model:
   ```bash
   python generate.py
   ```

---

## **🛠 Training Process**
### **1️⃣ Load Dataset**
```python
dataset = load_dataset("text", data_files={"train": file_path})
```
### **2️⃣ Split Dataset**
```python
split_dataset = dataset["train"].train_test_split(test_size=0.1)
```
### **3️⃣ Tokenize and Prepare Data**
```python
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
```
### **4️⃣ Define Training Arguments**
```python
training_args = TrainingArguments(output_dir="./gpt2-finetuned", num_train_epochs=3)
```
### **5️⃣ Train the Model**
```python
trainer.train()
```
### **6️⃣ Save and Load the Model**
```python
model.save_pretrained("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")
```
### **7️⃣ Generate Text**
```python
generator = pipeline("text-generation", model="./gpt2-finetuned", tokenizer="./gpt2-finetuned")
```

---

## **📜 Example Output**
```
Input: "The future of AI is"
Output: "The future of AI is rapidly evolving, with new advancements..."
```

---

## **📌 Notes**
- Make sure your dataset is **formatted properly**.
- Use **GPU (if available)** for faster training.
- Adjust `num_train_epochs` and `batch_size` for better results.

---

## **🤝 Contributing**
Feel free to fork this repo, make changes, and submit a pull request! 🚀  

---

## **📜 License**
This project is licensed under the **MIT License**.
