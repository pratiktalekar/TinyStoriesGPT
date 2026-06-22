# run_model.py
import os
import torch
import tiktoken
from huggingface_hub import hf_hub_download
from model import GPT, GPTConfig

# -------------------------------------------------
# 1. Config constants
# -------------------------------------------------
MODEL_FILENAME = "best_model_params.pt"
HF_REPO_ID = "pratiktalekar/tinystoriesgpt"

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
# 2. Lazy-loaded globals (saves memory at startup)
# -------------------------------------------------
_model = None
_enc = None
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = "cpu"  # force CPU on deployment (no GPU on Render)
    return _device


def _get_tokenizer():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("gpt2")
    return _enc


def _get_model():
    """Load model on first call, not at import time."""
    global _model
    if _model is not None:
        return _model

    device = _get_device()

    # Find or download model weights
    model_path = MODEL_FILENAME
    if not os.path.exists(model_path):
        print(f"📥 Local '{MODEL_FILENAME}' not found. Downloading from HuggingFace Hub...")
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"✅ Model downloaded to: {model_path}")
    else:
        print(f"✅ Using local model file: {model_path}")

    _model = GPT(config)
    _model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    _model.to(device)
    _model.eval()
    print("✅ Model loaded and ready for inference")
    return _model


# -------------------------------------------------
# 3. Text generation helper
# -------------------------------------------------
@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int = 100) -> str:
    """Generate text using the trained GPT model."""
    model = _get_model()
    enc = _get_tokenizer()
    device = _get_device()

    input_ids = enc.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...]  # add batch dimension
    y = model.generate(x, max_new_tokens=max_new_tokens)
    out = enc.decode(y[0].tolist())
    return out


# -------------------------------------------------
# 4. Example usage
# -------------------------------------------------
if __name__ == "__main__":
    prompt = "once upon a time"
    output = generate_text(prompt, max_new_tokens=100)
    print("Prompt:", prompt)
    print("\nGenerated text:\n", output)
