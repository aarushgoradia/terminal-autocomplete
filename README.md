# Terminal Autocomplete

A lightweight ML-powered CLI tool that autocompletes your terminal commands based on your past `.bash_history`. Built using PyTorch, LSTM, and a custom tokenizer. This project stemmed from the fact that at work I have to use a ton of commands (that I tend to forget)!



## Features

- LSTM-based next-token prediction
- Tokenizer, batching, vocab, and training pipeline
- Interactive CLI with `typer`
- Beautiful `rich` table output
- Reusable model + vocab loading
- CUDA and CPU-compatible



## Project Structure

``` bash
ml-terminal-autocomplete/
├── model/
│ ├── model_def.py # LSTM architecture
│ ├── dataset.py # Tokenization + batching
│ ├── train_model.py # Training loop
│ └── predict.py # Inference
├── cli/
│ └── main.py # Interactive CLI tool
├── data/
│ ├── bash_history.txt # Raw data
│ ├── tokenized_sequences.pkl
│ └── vocab.json
├── saved_models/
│ └── best_model.pt # Saved trained model
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ml-terminal-autocomplete.git
cd ml-terminal-autocomplete

# Set up virtual env (optional but recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Train the Model
```bash
python model/dataset.py        # Tokenize + save sequences
python model/train_model.py    # Train and save best model
```

## Run Interactive CLI
```bash
python -m main interactive start
```
Example:
```bash
>>> git ch
╭──────────────┬─────────────╮
│ Token        │ Probability │
├──────────────┼─────────────┤
│ checkout     │ 0.8235      │
│ cherry-pick  │ 0.0921      │
│ commit       │ 0.0544      │
╰──────────────┴─────────────╯
```

## Data Reference

Data is sources from [n12bash](https://github.com/TellinaTool/nl2bash)!

## Roadmap

This project is functional end-to-end, but there are several areas for improvement to increase usefulness, accuracy, and polish:

### Phase 1 – Core Functionality
- [x] Load and preprocess a user’s `bash_history`
- [x] Tokenize sequences (char-level)
- [x] Train an LSTM-based model to predict the next character
- [x] Build a prediction engine using the trained model
- [x] Implement a CLI with `typer` and styled output via `rich`
- [x] Allow interactive predictions from user-typed shell fragments

---

### Phase 2 – Model Improvement (In Progress)
- [ ] Switch to **token-level modeling** (predict next full token, not character)
- [ ] Improve dataset quality (real-world command history, fewer `<UNK>` tokens)
- [ ] Add greedy or beam search to complete full word predictions from char-level model (if char-level is kept)
- [ ] Add more training data (2k–5k lines) for better generalization
- [ ] Introduce dropout/regularization tuning for better performance
- [ ] Optionally try transformer-based architecture (`nn.Transformer`, `GPT2`, etc.)

---

### Phase 3 – Evaluation & Testing
- [ ] Add train/val/test split (e.g., 80/10/10)
- [ ] Report metrics during training (e.g., loss, accuracy)
- [ ] Add test-time evaluation: given real partial commands, does the model predict the correct token?
- [ ] Add per-epoch logging and loss visualization (e.g., `matplotlib`, `tensorboard`, or simple CLI output)

---

### Phase 4 – User Experience & Features
- [ ] Add shell-style **autocomplete** (TAB-key mimic, fuzzy match)
- [ ] Auto-complete full commands rather than just showing suggestions
- [ ] Add CLI option to show top-k completions as a single line or inline
- [ ] Create a `bash` or `zsh` plugin that calls the model for real-time shell autocompletion
- [ ] Make CLI installable via `pip` (`setup.py` or `pyproject.toml`)

---

### Phase 5 – Deployment & Distribution
- [ ] Add `pip` install support (e.g., `pip install ml-terminal-autocomplete`)
- [ ] Dockerize for easy use anywhere
- [ ] Publish demo video/gif in README
- [ ] Add usage examples in the README with screenshots
