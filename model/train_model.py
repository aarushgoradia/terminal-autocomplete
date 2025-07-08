import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from dataset import load_sequences, build_vocab, batchify
from model_def import LSTMNextTokenPredictor

# ==== Config ====
DATA_PATH = "data/tokenized_sequences.pkl"
VOCAB_PATH = "data/vocab.json"
MODEL_SAVE_PATH = "saved_models/best_model.pt"
BATCH_SIZE = 32
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ==== Load Data ====
print("[*] Loading tokenized sequences...")
sequences = load_sequences(DATA_PATH)

print("[*] Building vocabulary...")
vocab = build_vocab(sequences)
vocab_size = len(vocab)

# Optionally save vocab to JSON
os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)
with open(VOCAB_PATH, "w") as f:
    json.dump(vocab, f)

# ==== Create Model ====
model = LSTMNextTokenPredictor(
    vocab_size=vocab_size,
    embedding_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== Training Loop ====
print(f"[*] Starting training for {NUM_EPOCHS} epochs...")

best_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_inputs, batch_targets in batchify(sequences, vocab, batch_size=BATCH_SIZE):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)  # [batch_size, vocab_size]
        loss = loss_fn(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        print("New best model found. Saving...")
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        best_loss = avg_loss

print("[✓] Training complete.")
