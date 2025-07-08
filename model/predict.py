import torch
import torch.nn.functional as F
import json
import argparse

from model_def import LSTMNextTokenPredictor
from dataset import tokenize

# ==== Paths ====
MODEL_PATH = "saved_models/best_model.pt"
VOCAB_PATH = "data/vocab.json"

# ==== Load vocab ====
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
inv_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load Model Once ====
model = LSTMNextTokenPredictor(vocab_size=vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==== Prediction Function ====
def predict_next_token(prompt: str, top_k: int = 5):
    tokens = tokenize(prompt)
    token_ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)  # [1, vocab_size]
        probs = F.softmax(output, dim=1).squeeze()  # [vocab_size]

    topk_probs, topk_indices = torch.topk(probs, top_k)
    predictions = [(inv_vocab[idx.item()], prob.item()) for idx, prob in zip(topk_indices, topk_probs)]
    return predictions

# ==== CLI Entrypoint ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict next terminal token")
    parser.add_argument("--prompt", type=str, required=True, help="Partial command prompt, e.g., 'git ch'")
    parser.add_argument("--top_k", type=int, default=5, help="How many top predictions to return")
    args = parser.parse_args()

    preds = predict_next_token(args.prompt, top_k=args.top_k)
    print(f"\nPredictions for: '{args.prompt}'\n")
    for token, score in preds:
        print(f"{token:15s}  (prob: {score:.4f})")
