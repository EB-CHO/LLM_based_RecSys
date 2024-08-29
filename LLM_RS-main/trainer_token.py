import pandas as pd
import numpy as np
import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer
from input_prompt import prompt_generator

def load_tokens_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        tokens = json.load(file)
    return tokens

def preprocess(file_name):

    with open(file_name, "r") as f:
        dataset = json.load(f)

    json_file_path = 'special_tokens_map.json'
    tokens = load_tokens_from_json(json_file_path)

    bos_token = tokens["bos_token"]
    eos_token = tokens["eos_token"]
    pad_token = tokens["pad_token"]
    dataset_dict = {"text": []}

    random.shuffle(dataset) # 데이터를 무작위로 섞음
    for row in dataset:  
        mode = np.random.choice([0, 1])
        prompt_dict = prompt_gen.prompt_input(row, mode)
        if mode == 0:  # recommend 모드
            text = (f"{bos_token} Below is an instruction that describes a task, paired with an input that provides further context. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"## Instruction:\n{prompt_dict['instruction_recommend']}\n\n ## Input:\n{prompt_dict['input']} \n\n ## Response:\n{prompt_dict['output']} {eos_token}")
        
        elif mode == 1:  # generate 모드
            text = (f"{bos_token} Below is an instruction that describes a task, paired with an input that provides further context. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"## Instruction:\n{prompt_dict['instruction_generate']}\n\n ## Input:\n{prompt_dict['input']} \n\n ## Response:\n{prompt_dict['output']} {eos_token}")

        dataset_dict["text"].append(text)
    preprocessed_dataset = Dataset.from_dict(dataset_dict)
    
    return preprocessed_dataset

if __name__ == "__main__":

    restaurant_json = pd.read_json('restaurant_info.json') 
    prompt_gen = prompt_generator(restaurant_json)
    base_model = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    model = AutoModelForCausalLM.from_pretrained(base_model)
    new_model = "Llama3-Ko-3-8B-LLMRS"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    EOS_TOKEN = tokenizer.eos_token

    # training configs
    dataset_json = "final_dataset.json"
    dataset = preprocess(dataset_json)
    lr = 3e-6
    epochs = 10
    batch_per_device=6
    gradient_accumulation_steps = 8

    # training arguments
    training_args = TrainingArguments(
    output_dir="./save_checkpoints_llmrs",
    per_device_train_batch_size=batch_per_device,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=lr,
    logging_steps=5,
    num_train_epochs=epochs,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # eval_strategy="epoch",
    logging_first_step=True,
    )

    # trainer
    trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=1024,
    train_dataset=dataset,
    dataset_text_field="text",
    packing=True,
    )

    trainer.train()
    trainer.save_model(new_model)
