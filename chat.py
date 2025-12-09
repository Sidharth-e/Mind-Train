import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# --- CONFIGURATION ---
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_path = "Sidharth_AI_Model"

print(f"Loading base model: {base_model_name}...")
print("(This uses your RTX 3050 Ti VRAM)")

# --- LOAD MODEL ---
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# --- LOAD ADAPTER ---
print(f"Loading your custom training from: {adapter_path}...")
try:
    model = PeftModel.from_pretrained(model, adapter_path)
    print("✅ Success! Your custom Sidharth AI is loaded.")
except Exception as e:
    print(f"❌ Error loading adapter: {e}")
    exit()

# --- CUSTOM STOPPING CRITERIA ---
class StopOnTokens(StoppingCriteria):
    """Stop generation when specific tokens/patterns are detected"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Tokens that indicate end of response
        self.stop_tokens = [
            tokenizer.eos_token_id,
            tokenizer.encode("###", add_special_tokens=False)[0] if len(tokenizer.encode("###", add_special_tokens=False)) > 0 else None,
        ]
        self.stop_tokens = [t for t in self.stop_tokens if t is not None]
        
    def __call__(self, input_ids, scores, **kwargs):
        # Check if any of the last generated tokens are stop tokens
        if len(input_ids[0]) > 0:
            last_token = input_ids[0][-1].item()
            if last_token in self.stop_tokens:
                return True
        return False

def clean_response(text):
    """Clean up the model's response by removing training artifacts"""
    # Remove common training artifacts
    patterns_to_remove = [
        r'<\|end[_\s]*of[_\s]*sentence\|>',  # End of sentence token
        r'<\|end▁of▁sentence\|>',            # Unicode variant
        r'</s>',                              # EOS token
        r'<s>',                               # BOS token
        r'###\s*(Instruction|Response|Input):.*',  # Training format markers
        r'###.*$',                            # Any remaining ### markers
        r'\[INST\].*?\[/INST\]',              # Instruction markers
        r'Below is an instruction.*Response:',  # Alpaca prompt leakage
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove multiple spaces and clean up
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Take only the first complete response (stop at any new question marker)
    if '\n\n' in text:
        parts = text.split('\n\n')
        # Keep only the first meaningful paragraph
        text = parts[0].strip()
    
    return text

# --- CHAT SETUP ---
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
"""

model.eval()
stop_criteria = StoppingCriteriaList([StopOnTokens(tokenizer)])

print("\n" + "="*50)
print("🤖 Sidharth AI Ready. Type 'exit' to quit.")
print("="*50 + "\n")

while True:
    question = input("\033[1;32mYou:\033[0m ")
    if question.lower() in ["exit", "quit", "q"]:
        break

    # Format the input
    input_text = alpaca_prompt.format(question)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]

    # Generate (non-streaming for cleaner output)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,           # Shorter for concise answers
            temperature=0.3,              # Lower = more focused responses
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.3,       # Stronger repetition prevention
            no_repeat_ngram_size=3,       # Prevent repeating 3-grams
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stop_criteria,
            use_cache=True
        )
    
    # Decode only the new tokens (skip the prompt)
    generated_tokens = outputs[0][input_length:]
    raw_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up the response
    clean_text = clean_response(raw_response)
    
    # Display the cleaned response
    print(f"\033[1;36mAI:\033[0m {clean_text}\n")