import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel

# --- CONFIGURATION ---
# 1. The base model we trained on
base_model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"

# 2. Your trained adapter folder
adapter_path = "sidharth_backup_lora" 

# If your training crashed at the very end, use the checkpoint folder instead:
# adapter_path = "outputs/checkpoint-60" 

print(f"Loading base model: {base_model_name}...")
print("(This uses your RTX 3050 Ti VRAM)")

# --- LOAD MODEL ---
# Load the base model in 4-bit mode (low memory)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# --- LOAD YOUR DATA ---
print(f"Loading your custom training from: {adapter_path}...")
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print("✅ Success! Your custom Sidharth AI is loaded.")
except Exception as e:
    print(f"❌ Error loading adapter: {e}")
    print("Check if the folder name is correct.")
    exit()

# --- CHAT SETUP ---
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

# Enable faster inference
model.eval()

print("\n" + "="*50)
print("🤖 Sidharth AI Ready. Type 'exit' to quit.")
print("="*50 + "\n")

while True:
    question = input("\033[1;32mYou:\033[0m ") # Green text for user
    if question.lower() in ["exit", "quit", "q"]:
        break

    # Format the input
    input_text = alpaca_prompt.format(question)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # Stream the output so it looks like it's typing
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\033[1;36mAI:\033[0m ", end="") # Cyan text for AI
    
    # Generate
    with torch.no_grad():
        model.generate(
            **inputs, 
            streamer=streamer,
            max_new_tokens=256,
            temperature=0.7,  # Slightly higher for natural responses
            do_sample=True,   # Required for temperature to work
            top_p=0.9,        # Nucleus sampling for better quality
            repetition_penalty=1.2,  # Prevents repetition loops
            eos_token_id=tokenizer.eos_token_id,  # Stop at end token
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )
    print("\n")