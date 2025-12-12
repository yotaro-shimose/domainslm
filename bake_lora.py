from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model and LoRA adapters
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B")
lora_model: PeftModel = PeftModel.from_pretrained(
    base_model, "./qwen3-4b-sql-finetuned"
)

# Merge LoRA weights into base model
merged_model = lora_model.merge_and_unload()  # type: ignore

# Save merged model
merged_model.save_pretrained("./qwen3-4b-sql-merged")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
tokenizer.save_pretrained("./qwen3-4b-sql-merged")
