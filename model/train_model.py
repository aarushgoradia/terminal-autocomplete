import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from dataset import load_sequences, load_vocab, batchify
from model_def import LSTMNextTokenPredictor

# ==== Config ====
DATA_PATH = "data/tokenized_sequences.pkl"
VOCAB_PATH = "data/vocab.json"
MODEL_SAVE_PATH = "saved_models/best_model.pt"
CONFIG_SAVE_PATH = "saved_models/config.json"

BATCH_SIZE = 32
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ==== Load Data ====
print("[*] Loading tokenized sequences...")
sequences = load_sequences(DATA_PATH)

print("[*] Loading vocabulary...")
token2id, _ = load_vocab(VOCAB_PATH)
vocab_size = len(token2id)

# ==== Save config ====
os.makedirs(os.path.dirname(CONFIG_SAVE_PATH), exist_ok=True)
with open(CONFIG_SAVE_PATH, "w") as f:
    json.dump({
        "embedding_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": 1
    }, f)

# ==== Initialize Model ====
model = LSTMNextTokenPredictor(
    vocab_size=vocab_size,
    embedding_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== Training ====
print(f"[*] Training model for {NUM_EPOCHS} epochs...")
best_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    batches = batchify(sequences, token2id, batch_size=BATCH_SIZE)

    for batch_inputs, batch_targets in batches:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(batches)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        print("[✓] New best model — saving.")
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        best_loss = avg_loss

print("[✓] Training complete.")
