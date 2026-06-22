# run_model.py
import os
import torch
import tiktoken
from huggingface_hub import hf_hub_download
from model import GPT, GPTConfig

# -------------------------------------------------
# 1. Load tokenizer (same as used in training)
# -------------------------------------------------
enc = tiktoken.get_encoding("gpt2")

# -------------------------------------------------
# 2. Define model config (extracted from your notebook)
# -------------------------------------------------
config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=128,       # or whatever context size you're training with
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

# -------------------------------------------------
# 3. Initialize model and load trained weights
#    - Uses local file if present
#    - Otherwise downloads from HuggingFace Hub
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FILENAME = "best_model_params.pt"
HF_REPO_ID = "pratiktalekar/tinystoriesgpt"

model_path = MODEL_FILENAME
if not os.path.exists(model_path):
    print(f"📥 Local '{MODEL_FILENAME}' not found. Downloading from HuggingFace Hub...")
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME,
        token=os.environ.get("HF_TOKEN"),  # needed if repo is private
    )
    print(f"✅ Model downloaded to: {model_path}")
else:
    print(f"✅ Using local model file: {model_path}")

model = GPT(config)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -------------------------------------------------
# 4. Text generation helper
# -------------------------------------------------
@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    """Generate text using the trained GPT model."""
    input_ids = enc.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]  # add batch dimension
    y = model.generate(x, max_new_tokens=max_new_tokens)
    out = enc.decode(y[0].tolist())
    return out

# -------------------------------------------------
# 5. Example usage
# -------------------------------------------------
if __name__ == "__main__":
    prompt = "once upon a time"
    output = generate_text(prompt, max_new_tokens=100)
    print("Prompt:", prompt)
    print("\nGenerated text:\n", output)
