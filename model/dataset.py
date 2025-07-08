import os
import pickle
import random
import torch
from collections import Counter

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"

def load_raw_history(filepath):
    """
    Load lines from a bash history file, stripping empty or comment lines.
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines

def tokenize(line):
    """
    Simple whitespace tokenizer.
    """
    return line.split()

def generate_sequences(lines):
    """
    Converts lines of commands into (input, target) training pairs.
    Each token in a command becomes part of an input sequence with the next token as target.
    """
    sequences = []
    for line in lines:
        tokens = tokenize(line)
        if not tokens:
            continue
        tokens.append(EOS_TOKEN)  # End of sequence token
        for i in range(1, len(tokens)):
            input_seq = tokens[:i]
            target_token = tokens[i]
            sequences.append((input_seq, target_token))
    return sequences

def save_sequences(sequences, output_path):
    """
    Save tokenized sequences to a binary .pkl file.
    """
    with open(output_path, "wb") as f:
        pickle.dump(sequences, f)

def load_sequences(input_path):
    """
    Load tokenized sequences from a .pkl file.
    """
    with open(input_path, "rb") as f:
        return pickle.load(f)

def pad_sequence(seq, max_len, pad_token_id):
    return seq + [pad_token_id] * (max_len - len(seq))

def build_vocab(sequences, min_freq=1):
    """
    Builds vocab dicts from token sequences.
    """
    counter = Counter()
    for input_seq, target in sequences:
        counter.update(input_seq)
        counter.update([target])

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, EOS_TOKEN: 2}
    idx = len(vocab)

    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1

    return vocab

def batchify(sequences, vocab, batch_size=32):
    """
    Yields batches of padded tokenized input sequences and target tokens.
    """
    random.shuffle(sequences)
    pad_token_id = vocab[PAD_TOKEN]
    unk_token_id = vocab[UNK_TOKEN]

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        input_seqs = []
        target_tokens = []

        for input_seq, target in batch:
            input_ids = [vocab.get(tok, unk_token_id) for tok in input_seq]
            target_id = vocab.get(target, unk_token_id)
            input_seqs.append(input_ids)
            target_tokens.append(target_id)

        max_len = max(len(seq) for seq in input_seqs)
        padded_inputs = [pad_sequence(seq, max_len, pad_token_id) for seq in input_seqs]

        yield torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)


if __name__ == "__main__":
    raw_path = "data/bash_history.txt"
    processed_path = "data/tokenized_sequences.pkl"

    print("[*] Loading raw history...")
    lines = load_raw_history(raw_path)

    print(f"[*] Tokenizing {len(lines)} lines...")
    sequences = generate_sequences(lines)

    print(f"[*] Saving {len(sequences)} input-target pairs to {processed_path}")
    save_sequences(sequences, processed_path)

    # Optional test
    print("[*] Sample sequences:")
    for i in range(5):
        print("Input :", sequences[i][0])
        print("Target:", sequences[i][1])
        print("---")
