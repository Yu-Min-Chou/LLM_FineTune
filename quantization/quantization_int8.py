import os

import torch
import torch.nn as nn
import transformers
import bitsandbytes as bnb
from datasets import load_dataset
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b")

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", load_in_8bit=True, device_map="auto")
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            max_steps=200,
            learning_rate=2e-4,
            fp16=False,
            logging_steps=1,
            output_dir="outputs",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

if __name__ == "__main__":
    main()