import torch
import torch.nn as nn

class LSTMNextTokenPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.1):
        super(LSTMNextTokenPredictor, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        _, (hidden, _) = self.lstm(embedded)  # hidden: [num_layers, batch_size, hidden_dim]
        last_hidden = hidden[-1]              # [batch_size, hidden_dim]
        output = self.fc(last_hidden)         # [batch_size, vocab_size]
        return output
