import os
import re
import json
import pickle
from collections import Counter
from typing import List, Tuple

# ====== Tokenization Mode ======
TOKEN_LEVEL = "char"  # change to "word" or "char"

# ====== Tokenizer ======
def tokenize(line: str) -> List[str]:
    if TOKEN_LEVEL == "char":
        return list(line.strip()) + ["<EOS>"]
    else:
        return line.strip().split() + ["<EOS>"]

# ====== Sequence Generator ======
def generate_sequences(lines: List[str], add_prefixes: bool = True, max_prefixes_per_token: int = 3) -> List[Tuple[List[str], str]]:
    """
    Generate input→target token pairs.
    If char-level, adds all character-based sequence pairs.
    If word-level, optionally includes token prefixes like 'git ch' → 'checkout'.
    """
    sequences = []
    for line in lines:
        tokens = tokenize(line)
        for i in range(1, len(tokens)):
            input_seq = tokens[:i]
            target = tokens[i]
            sequences.append((input_seq, target))

            # Add prefixes only for word-level
            if TOKEN_LEVEL == "word" and add_prefixes and 1 < len(target) <= 10 and re.match(r"^[a-zA-Z0-9\-_/]+$", target):
                for j in range(2, min(len(target), 6)):
                    prefix = target[:j]
                    sequences.append((input_seq[:-1] + [prefix], target))
    return sequences

# ====== Vocab Builder ======
def build_vocab(sequences: List[Tuple[List[str], str]], min_freq: int = 1):
    counter = Counter()
    for input_seq, target in sequences:
        counter.update(input_seq)
        counter.update([target])
    special_tokens = ["<PAD>", "<UNK>", "<EOS>"]
    all_tokens = special_tokens + sorted(set(tok for tok, count in counter.items() if count >= min_freq and tok not in special_tokens))
    token2id = {tok: idx for idx, tok in enumerate(all_tokens)}
    id2token = {idx: tok for tok, idx in token2id.items()}
    return token2id, id2token

def save_vocab(token2id: dict, path: str):
    with open(path, "w") as f:
        json.dump(token2id, f)

def load_vocab(path: str):
    with open(path, "r") as f:
        token2id = json.load(f)
    id2token = {v: k for k, v in token2id.items()}
    return token2id, id2token

# ====== Sequence Save/Load ======
def save_sequences(sequences: List[Tuple[List[str], str]], path: str):
    with open(path, "wb") as f:
        pickle.dump(sequences, f)

def load_sequences(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

# ====== Batching ======
def batchify(sequences: List[Tuple[List[str], str]], token2id: dict, batch_size: int = 32, max_len: int = 64):
    import torch
    pad_id = token2id.get("<PAD>", 0)
    unk_id = token2id.get("<UNK>", 1)

    def encode(seq):
        return [token2id.get(tok, unk_id) for tok in seq]

    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        input_seqs = [encode(seq[:max_len]) for seq, _ in batch]
        targets = [token2id.get(target, unk_id) for _, target in batch]
        max_seq_len = max(len(seq) for seq in input_seqs)
        padded = [seq + [pad_id] * (max_seq_len - len(seq)) for seq in input_seqs]

        input_tensor = torch.tensor(padded, dtype=torch.long)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        batches.append((input_tensor, target_tensor))
    return batches

# ====== CLI Entrypoint ======
if __name__ == "__main__":
    raw_path = "data/bash_history.txt"
    seq_path = "data/tokenized_sequences.pkl"
    vocab_path = "data/vocab.json"

    print(f"[*] Using {TOKEN_LEVEL}-level tokenization")
    print(f"[*] Loading data from {raw_path}")
    with open(raw_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print("[*] Generating sequences...")
    sequences = generate_sequences(lines, add_prefixes=True)
    save_sequences(sequences, seq_path)
    print(f"[*] Saved {len(sequences)} sequences to {seq_path}")

    print("[*] Building vocab...")
    token2id, id2token = build_vocab(sequences)
    save_vocab(token2id, vocab_path)
    print(f"[*] Saved vocab with {len(token2id)} tokens to {vocab_path}")

    print("[*] Sample examples:")
    for i in range(5):
        print("Input:", sequences[i][0])
        print("Target:", sequences[i][1])
