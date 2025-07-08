import torch
import torch.nn.functional as F
import json
import argparse
import os

from .model_def import LSTMNextTokenPredictor
from .dataset import tokenize

# ==== Paths ====
MODEL_PATH = "saved_models/best_model.pt"
VOCAB_PATH = "data/vocab.json"
CONFIG_PATH = "saved_models/config.json"

# ==== Load Vocab ====
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

# ==== Load Config ====
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    embedding_dim = config.get("embedding_dim", 128)
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 1)
else:
    embedding_dim, hidden_dim, num_layers = 128, 256, 1

# ==== Setup Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Model ====
model = LSTMNextTokenPredictor(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==== Predict ====
def predict_next_token(prompt: str, top_k: int = 5):
    tokens = tokenize(prompt)
    token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)  # [1, vocab_size]
        probs = F.softmax(output, dim=1).squeeze()

    topk_probs, topk_indices = torch.topk(probs, top_k)
    predictions = [(inv_vocab[idx.item()], prob.item()) for idx, prob in zip(topk_indices, topk_probs)]
    return predictions

# ==== CLI Entrypoint ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next terminal token (character-level)")
    parser.add_argument("--prompt", type=str, required=True, help="e.g., 'git ch'")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    preds = predict_next_token(args.prompt, top_k=args.top_k)
    print(f"\nPredictions for: '{args.prompt}'\n")
    for token, score in preds:
        print(f"{token:15s}  (prob: {score:.4f})")
