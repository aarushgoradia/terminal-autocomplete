# 🧠 ML Terminal Autocomplete

A lightweight ML-powered CLI tool that autocompletes your terminal commands based on your past `.bash_history`. Built using PyTorch, LSTM, and a custom tokenizer.

---

## 🚀 Features

- 🧠 LSTM-based next-token prediction
- 📦 Tokenizer, batching, vocab, and training pipeline
- ⚙️ Interactive CLI with `typer`
- 🎨 Beautiful `rich` table output
- 💾 Reusable model + vocab loading
- ⚡ CUDA and CPU-compatible

---

## 📁 Project Structure

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

## 🛠️ Setup

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

## 🧠 Train the Model
```bash
python model/dataset.py        # Tokenize + save sequences
python model/train_model.py    # Train and save best model
```

## 🧪 Run Interactive CLI
```bash
python cli/main.py interactive
```
Example:
>>> git ch
╭──────────────┬─────────────╮
│ Token        │ Probability │
├──────────────┼─────────────┤
│ checkout     │ 0.8235      │
│ cherry-pick  │ 0.0921      │
│ commit       │ 0.0544      │
╰──────────────┴─────────────╯

## 📄 License
MIT License

## ✨ Coming Soon
🧩 Shell tab completion integration
