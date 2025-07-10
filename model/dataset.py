import os
import re
import json
import pickle
import argparse
from collections import Counter
from typing import List, Tuple

# ====== Tokenizer ======
def tokenize(line: str, level: str) -> List[str]:
    line = line.strip()
    if not line:
        return []
    if level == "char":
        return list(line) + ["<EOS>"]
    return line.split() + ["<EOS>"]

# ====== Sequence Generator ======
def generate_sequences(lines: List[str], level: str, add_prefixes: bool = True) -> List[Tuple[List[str], str]]:
    sequences = []
    for line in lines:
        tokens = tokenize(line, level)
        for i in range(1, len(tokens)):
            input_seq = tokens[:i]
            target = tokens[i]
            sequences.append((input_seq, target))

            # Augment with prefixes for word-level
            if level == "word" and add_prefixes and 1 < len(target) <= 10 and re.match(r"^[a-zA-Z0-9\-_/.]+$", target):
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
    parser = argparse.ArgumentParser(description="Generate tokenized training data")
    parser.add_argument("--level", choices=["char", "word"], default="char", help="Tokenization level")
    parser.add_argument("--input", default="data/bash_history.txt", help="Path to bash history file")
    parser.add_argument("--out_seq", default=None, help="Output path for tokenized sequences")
    parser.add_argument("--out_vocab", default=None, help="Output path for vocab JSON")
    parser.add_argument("--add-prefixes", action="store_true", help="Add prefix-augmented sequences (only for word-level)")
    args = parser.parse_args()

    level = args.level
    print(f"[*] Using {level}-level tokenization")
    print(f"[*] Loading from {args.input}")

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    print("[*] Generating sequences...")
    sequences = generate_sequences(lines, level=level, add_prefixes=args.add_prefixes)

    out_seq = args.out_seq or f"data/{level}_sequences.pkl"
    out_vocab = args.out_vocab or f"data/{level}_vocab.json"

    save_sequences(sequences, out_seq)
    print(f"[*] Saved {len(sequences)} sequences to {out_seq}")

    print("[*] Building vocab...")
    token2id, id2token = build_vocab(sequences)
    save_vocab(token2id, out_vocab)
    print(f"[*] Saved vocab with {len(token2id)} tokens to {out_vocab}")

    print("[*] Sample examples:")
    for i in range(5):
        print("Input:", sequences[i][0])
        print("Target:", sequences[i][1])
