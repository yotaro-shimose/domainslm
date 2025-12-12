from pathlib import Path

import torch
import weave
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig

from domainslm.spider.sft_sample import SFTDataset

# Load reflection results
train_path = Path("data/reflection_train.json")
# Wait for file to exist if running concurrently (or just assume it will exist)
if not train_path.exists():
    raise FileNotFoundError(
        f"{train_path} not found. Please run generate_reflections.py first."
    )

reflections = SFTDataset.model_validate_json(train_path.read_text())
print(f"Loaded {len(reflections.samples)} training reflection results")

eval_path = Path("data/reflection_eval.json")

# Model configuration
max_seq_length = 4096 * 5
model_name = "Qwen/Qwen3-4B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

# Load model with quantization config for efficiency
use_quantization = False  # Set to True to use 4-bit quantization

if use_quantization:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()  # type: ignore


# Convert reflections to dataset
dataset_dict: list[dict] = [r.as_chatml() for r in reflections.samples]  # type:ignore
dataset = reflections.as_prompt_completion()
print(f"Prepared dataset with {len(dataset)} examples")

# Determine if bf16 is supported
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# Training arguments - optimized for 80GB GPU memory
training_args = SFTConfig(
    output_dir="./outputs",
    completion_only_loss=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # Reduced from 4 to 2 (still same effective batch size: 8*2*2=32)
    warmup_steps=10,  # Slightly increased for more stable training
    num_train_epochs=2,  # Train for 2 epochs
    learning_rate=4e-4,
    logging_steps=1,
    optim="adamw_torch_fused",  # Fused optimizer for better performance
    weight_decay=0.01,
    max_length=max_seq_length,
    lr_scheduler_type="cosine",  # Cosine scheduler often works better than linear
    seed=3407,
    save_strategy="epoch",
    fp16=not use_bf16,
    bf16=use_bf16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # Multi-GPU configuration for 2 devices
    ddp_find_unused_parameters=False,
    dataloader_num_workers=8,  # Increased for better data loading throughput
    dataset_num_proc=4,  # Increased preprocessing parallelism
    remove_unused_columns=False,
    packing=False,
    # Additional optimizations
    dataloader_pin_memory=True,  # Pin memory for faster GPU transfers
    dataloader_prefetch_factor=2,  # Prefetch batches
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Start training
print("\nStarting training...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./qwen3-4b-sql-finetuned")
tokenizer.save_pretrained("./qwen3-4b-sql-finetuned")
print("\nModel saved to ./qwen3-4b-sql-finetuned")
