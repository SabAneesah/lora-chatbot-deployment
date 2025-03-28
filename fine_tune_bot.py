#!/usr/bin/env python3
# finetune_fashion_lm.py
import torch
import json
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import pandas as pd
from faker import Faker  # For generating synthetic data

# --------------------
# Configurations
# --------------------
class Config:
    MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # or "mistralai/Mixtral-8x7B-v0.1"
    DATASET_PATH = "./fashion_qa.jsonl"  # Path to save/load dataset
    OUTPUT_DIR = "./mistral-fashion-lora"
    LORA_RANK = 8
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    EPOCHS = 3
    SYNTHETIC_SAMPLES = 1000  # Number of synthetic examples to generate

# --------------------
# Synthetic Data Generation
# --------------------
def generate_synthetic_data(num_samples: int) -> Dataset:
    """Generate fashion Q&A pairs using Faker and templates"""
    fake = Faker()
    data = []
    
    genders = ["Male", "Female", "Non-binary"]
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    styles = ["Casual", "Formal", "Business Casual", "Bohemian", "Streetwear"]
    budgets = ["Low", "Medium", "High"]
    
    for _ in range(num_samples):
        gender = fake.random_element(genders)
        season = fake.random_element(seasons)
        style = fake.random_element(styles)
        budget = fake.random_element(budgets)
        
        # Template-based generation
        occasion = fake.random_element([
            "work", "date night", "wedding", "beach party", 
            "job interview", "gala event", "weekend brunch"
        ])
        
        instruction = f"Recommend a {occasion} outfit for {gender.lower()}"
        input_text = f"Season: {season}, Style: {style}, Budget: {budget}"
        
        # Output generation with template
        outfit = {
            "Spring": f"light {fake.color_name()} {fake.random_element(['blazer', 'cardigan'])}",
            "Summer": f"{fake.color_name()} {fake.random_element(['linen shirt', 'sundress'])}",
            "Fall": f"{fake.color_name()} {fake.random_element(['knit sweater', 'corduroy pants'])}",
            "Winter": f"wool {fake.random_element(['coat', 'turtleneck'])}"
        }[season]
        
        accessories = {
            "Low": fake.random_element(["canvas tote", "simple sneakers"]),
            "Medium": fake.random_element(["leather belt", "ankle boots"]),
            "High": fake.random_element(["designer handbag", "luxury watch"])
        }[budget]
        
        output = (
            f"Wear a {outfit} with {fake.color_name()} {fake.random_element(['pants', 'skirt'])}. "
            f"Accessorize with {accessories}. {fake.random_element(['Add minimal jewelry.', 'Complete with sunglasses.'])}"
        )
        
        data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })
    
    # Save to JSONL
    with open(Config.DATASET_PATH, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    return Dataset.from_list(data)

# --------------------
# Dataset Preparation
# --------------------
def load_or_create_dataset(file_path: str) -> Dataset:
    """Load dataset or generate synthetic data if not found"""
    try:
        # Load dataset with explicit split handling
        dataset = load_dataset("json", data_files=file_path)["train"]
        print(f"✅ Loaded dataset with {len(dataset)} samples")
        
        # Correct validation check
        required_keys = {"instruction", "input", "output"}
        if not all(key in dataset.features for key in required_keys):
            raise ValueError(f"Dataset must contain {required_keys} columns")
            
    except FileNotFoundError:
        print("🛠️ Generating synthetic dataset...")
        dataset = generate_synthetic_data(Config.SYNTHETIC_SAMPLES)
        print(f"✨ Generated {len(dataset)} synthetic samples")
    
    return dataset.train_test_split(test_size=0.1, shuffle=True)
    
# --------------------
# Tokenization
# --------------------
def tokenize_data(tokenizer, examples):
    """Tokenize with proper padding handling"""
    # Ensure tokenizer is properly configured
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format prompts
    prompts = [
        f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n"
        for ins, inp in zip(examples["instruction"], examples["input"])
    ]
    
    # Tokenize
    model_inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length"  # Requires pad_token
    )
    
    # Tokenize labels separately
    labels = tokenizer(
        examples["output"],
        truncation=True,
        max_length=512,
        padding="max_length"
    ).input_ids
    
    model_inputs["labels"] = labels
    return model_inputs

# --------------------
# LoRA Finetuning
# --------------------
def finetune_model(train_dataset, eval_dataset):
    # Quantization Config (4-bit for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load Model with DDP for multi-GPU
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        quantization_config=None,
        device_map="auto",
        torch_dtype=torch.float16,
        #attn_implementation="flash_attention_2"  # If available
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_NAME,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=Config.LORA_RANK,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=Config.LEARNING_RATE,
        num_train_epochs=Config.EPOCHS,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # Trainer with packed sequences
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        packing=True,  # Efficient sequence packing
        max_seq_length=512,
        dataset_text_field="instruction",  # Field to use for packing
    )
    
    print("🚀 Starting training...")
    trainer.train()
    
    # Save only adapters to save space
    model.save_pretrained(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    print(f"💾 Model saved to {Config.OUTPUT_DIR}")
    
    return model, tokenizer

# --------------------
# Inference
# --------------------
def generate_outfit(model, tokenizer, prompt: str, max_length=200):
    """Generate outfit recommendation with temperature sampling"""
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    result = pipe(
        prompt,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        num_return_sequences=1
    )
    
    return result[0]["generated_text"]

# --------------------
# Main Execution
# --------------------
if __name__ == "__main__":
    # Step 1: Data Preparation
    dataset = load_or_create_dataset(Config.DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    # Step 2: Tokenization
    tokenized_data = dataset.map(
        lambda x: tokenize_data(tokenizer, x),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Step 3: Finetuning
    model, tokenizer = finetune_model(
        tokenized_data["train"],
        tokenized_data["test"]
    )
    
    # Step 4: Interactive Testing
    while True:
        try:
            gender = input("Enter gender (Male/Female/Non-binary): ")
            season = input("Enter season (Spring/Summer/Fall/Winter): ")
            style = input("Enter style (Casual/Formal/etc.): ")
            
            prompt = (
                f"### Instruction:\nRecommend a {style.lower()} outfit\n\n"
                f"### Input:\nGender: {gender}, Season: {season}\n\n"
                f"### Response:\n"
            )
            
            print("\n🧥 Recommended Outfit:")
            print(generate_outfit(model, tokenizer, prompt))
            print("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break