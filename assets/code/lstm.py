
!pip install "numpy<2" # needed to install torchtext in colab
!pip install portalocker
!pip install torchtext==0.16.2
import string
import pandas as pd
!pip install portalocker
import torch
import torchtext
import portalocker

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import clear_output
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

PATH_TO_TRAIN_DATA = 'train.csv'
df = pd.read_csv(PATH_TO_TRAIN_DATA)
df.head()

df['review'] =  df['positive'] + ' ' + df['negative']
df_train, df_test = train_test_split(df, random_state=1412, train_size = 0.75)
tokenizer = get_tokenizer('spacy', language="en_core_web_sm")

def yield_tokens(data_iter):
    punctuation = set(string.punctuation)
    custom_stopwords = STOP_WORDS - {'no', 'not', 'with'}  # keep these words

    for text in data_iter:
        tokens = tokenizer(text.lower())
        filtered = [
            token for token in tokens
            if token not in punctuation and token not in custom_stopwords
        ]
        yield filtered

vocab = build_vocab_from_iterator(yield_tokens(df_train['review']),
                                  specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x.lower()))
label_pipeline = lambda x: float(x)

MAX_LENGTH = 300

def collate_batch(batch):
    text_list, label_list, lengths = [], [], []
    for text, score in batch:
        processed_text = text_pipeline(text[:MAX_LENGTH])
        text_list.append(torch.LongTensor(processed_text))
        label_list.append(label_pipeline(score))
        lengths.append(len(processed_text))

    text_list = pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.float32)
    return text_list.to(device), label_list.to(device), lengths

batch_size = 64
train_dataset = list(zip(df_train['review'].tolist(), df_train['score'].tolist()))
test_dataset = list(zip(df_test['review'].tolist(), df_test['score'].tolist()))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_batch)

class TextRegressor(nn.Module):
    def __init__(self, num_embeddings, embedding_size,
                 hidden_size, num_layers, padding_idx=0):
        super(TextRegressor, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, 1)
        self.dropout = nn.Dropout(0.7)


    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))
        pack_out = nn.utils.rnn.pack_padded_sequence(embedded, lengths,
                                                     batch_first=True, enforce_sorted=False).to(device)
        _, (last_hidden, last_c) = self.lstm(embedded)
        hidden = torch.cat([last_hidden[-2], last_hidden[-1]], dim=1)
        return self.linear(hidden).squeeze()

num_embeddings = len(vocab)
embedding_size = 300
hidden_size = 80
num_layers = 2

def train_epoch(
    model, data_loader, optimizer, criterion, return_losses=False, device="cuda:0",):
    model = model.to(device).train()
    total_loss = 0
    num_batches = 0
    all_losses = []
    total_predictions = np.array([])#.reshape((0, ))
    total_labels = np.array([])#.reshape((0, ))
    for x, y, lengths in data_loader:

        x = x.to(device)
        y = y.to(device).float()

        preds = model(x, lengths)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.detach().item()
        total_predictions = np.append(total_predictions, preds.cpu().detach().numpy())
        total_labels = np.append(total_labels, y.cpu().detach().numpy())
        num_batches += 1
        all_losses.append(loss.detach().item())

    # Compute regression metrics
    metrics = {
        "loss": total_loss / num_batches,
        "mae": np.mean(np.abs(total_predictions - total_labels)),
        "rmse": np.sqrt(np.mean((total_predictions - total_labels) ** 2)),
    }

    if return_losses:
        return metrics, all_losses
    else:
        return metrics

@torch.no_grad()
def validate(model, data_loader, criterion, device="cuda:0"):
    model = model.eval()
    total_loss = 0
    num_batches = 0
    total_predictions = np.array([])
    total_labels = np.array([])
    for x, y, lengths in data_loader:
        x = x.to(device)
        y = y.to(device).float()
        preds = model(x, lengths)
        loss = criterion(preds, y)

        total_loss += loss.detach().item()
        total_predictions = np.append(total_predictions, preds.cpu().detach().numpy())
        total_labels = np.append(total_labels, y.cpu().detach().numpy())
        num_batches += 1
    metrics = {
        "loss": total_loss / num_batches,
        "mae": np.mean(np.abs(total_predictions - total_labels)),
        "rmse": np.sqrt(np.mean((total_predictions - total_labels) ** 2)),
    }
    return metrics

device="cuda:0"
model = TextRegressor(num_embeddings, embedding_size,
                 hidden_size, num_layers, padding_idx=vocab['<pad>']).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.L1Loss()

train_accuracy, train_loss = [], []
val_accuracy, val_loss = [], []

for i in range(40):
    train_metrics = train_epoch(model, train_dataloader, criterion=criterion, optimizer=optimizer, device=device)
    val_metrics = validate(model, test_dataloader, criterion=criterion, device=device)

    train_accuracy.append(train_metrics["mae"])
    train_loss.append(train_metrics["loss"])
    val_accuracy.append(val_metrics["mae"])
    val_loss.append(val_metrics["loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0].twinx()
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Train MAE')
    ax.set_ylabel('Train MAE')
    ax.plot(train_accuracy, color="r", label='Train MAE')
    plt.legend()

    ax = axes[1].twinx()
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Val MAE')
    ax.set_ylabel('Val MAE')
    axes[1].plot(val_loss, color="b", label='Val loss')
    plt.legend()
    ax.plot(val_accuracy, color="r", label='Val MAE')
    plt.legend()

    fig.tight_layout()  # To ensure the right y-label is not slightly clipped
    plt.show()

    clear_output(wait=True)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention_query = nn.Parameter(torch.randn(hidden_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_outputs):
        # lstm_outputs: (batch_size, seq_len, hidden_dim)
        batch_size = lstm_outputs.size(0)
        query_repeated = self.attention_query.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)
        scores = torch.bmm(lstm_outputs, query_repeated).squeeze(2)
        attn_weights = self.softmax(scores)
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1) # (batch, hidden_dim)
        return context_vector, attn_weights

class LSTMAttentionHybrid(nn.Module):
    def __init__(self, num_embeddings, embedding_size,
                 hidden_size, num_layers, padding_idx=0):
        super(LSTMAttentionHybrid, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True,
                            num_layers=num_layers, bidirectional=True)

        self.attention = Attention(hidden_size * 2) # Input is 2 * hidden_size

        # Input is (hidden_size * 2) from simple LSTM + (hidden_size * 2) from Attention
        self.linear = nn.Linear(hidden_size * 4, 1)

        self.dropout = nn.Dropout(0.85)


    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))

        pack_out = nn.utils.rnn.pack_padded_sequence(embedded, lengths,
                                        batch_first=True, enforce_sorted=False)

        # need both the full sequence (lstm_outputs) and the final hidden state (last_hidden)
        lstm_outputs, (last_hidden, last_c) = self.lstm(pack_out)

        # Get the final hidden state from the last layer (batch, hidden_size * 2)
        hidden_summary = torch.cat([last_hidden[-2], last_hidden[-1]], dim=1)

        unpacked_outputs, _ = nn.utils.rnn.pad_packed_sequence(lstm_outputs, batch_first=True)
        context_vector, _ = self.attention(unpacked_outputs)

        # (batch, (hidden_size * 2) + (hidden_size * 2)) = (batch, hidden_size * 4)
        combined_signals = torch.cat([hidden_summary, context_vector], dim=1)
        combined_signals = self.dropout(combined_signals)
        return self.linear(combined_signals).squeeze()

def train_epoch(model, data_loader, optimizer, criterion, device="cuda:0"):
    model = model.to(device).train()
    total_loss = 0
    num_batches = 0
    total_predictions = np.array([])
    total_labels = np.array([])

    for text, y, lengths in data_loader:
        y = y.float()
        preds = model(text, lengths)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()
        total_predictions = np.append(total_predictions, preds.cpu().detach().numpy())
        total_labels = np.append(total_labels, y.cpu().detach().numpy())
        num_batches += 1

    metrics = {
        "loss": total_loss / (num_batches or 1),
        "mae": np.mean(np.abs(total_predictions - total_labels)),
        "rmse": np.sqrt(np.mean((total_predictions - total_labels) ** 2)),
    }
    return metrics

@torch.no_grad()
def validate(model, data_loader, criterion, device="cuda:0"):
    model = model.eval()
    total_loss = 0
    num_batches = 0
    total_predictions = np.array([])
    total_labels = np.array([])

    for text, y, lengths in data_loader:
        y = y.float()
        preds = model(text, lengths)
        loss = criterion(preds, y)

        total_loss += loss.detach().item()
        total_predictions = np.append(total_predictions, preds.cpu().detach().numpy())
        total_labels = np.append(total_labels, y.cpu().detach().numpy())
        num_batches += 1

    metrics = {
        "loss": total_loss / (num_batches or 1),
        "mae": np.mean(np.abs(total_predictions - total_labels)),
        "rmse": np.sqrt(np.mean((total_predictions - total_labels) ** 2)),
    }
    return metrics

num_embeddings = len(vocab)
embedding_size = 300
hidden_size = 80
num_layers = 2
pad_idx = vocab['<pad>']

model = LSTMAttentionHybrid(
    num_embeddings, embedding_size,
    hidden_size, num_layers, padding_idx=pad_idx
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
criterion = nn.L1Loss()

print("Starting Training with LSTMAttentionHybrid...")
train_mae_history, train_loss_history = [], []
val_mae_history, val_loss_history = [], []
N_EPOCHS = 30

for epoch in range(N_EPOCHS):
    train_metrics = train_epoch(model, train_dataloader, criterion=criterion, optimizer=optimizer, device=device)
    val_metrics = validate(model, test_dataloader, criterion=criterion, device=device)

    train_mae_history.append(train_metrics["mae"])
    train_loss_history.append(train_metrics["loss"])
    val_mae_history.append(val_metrics["mae"])
    val_loss_history.append(val_metrics["loss"])

    clear_output(wait=True)
    print(f'Epoch: {epoch+1:02}/{N_EPOCHS}')
    print(f'\tTrain Loss: {train_metrics["loss"]:.3f} | Train MAE: {train_metrics["mae"]:.3f}')
    print(f'\t Val. Loss: {val_metrics["loss"]:.3f} |  Val. MAE: {val_metrics["mae"]:.3f}')

print("Training Finished.")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1_loss = axes[0]
ax1_acc = ax1_loss.twinx()
ax1_loss.set_xlabel('Epochs')
ax1_loss.set_ylabel('Loss', color="b")
ax1_acc.set_ylabel('MAE', color="r")
ax1_loss.plot(train_loss_history, color="b", label='Train loss')
ax1_acc.plot(train_mae_history, color="r", label='Train MAE')
ax1_loss.set_title("Training Metrics")
lines1, labels1 = ax1_loss.get_legend_handles_labels()
lines2, labels2 = ax1_acc.get_legend_handles_labels()
ax1_loss.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

ax2_loss = axes[1]
ax2_acc = ax2_loss.twinx()
ax2_loss.set_xlabel('Epochs')
ax2_loss.set_ylabel('Loss', color="b")
ax2_acc.set_ylabel('MAE', color="r")
ax2_loss.plot(val_loss_history, color="b", label='Val loss')
ax2_acc.plot(val_mae_history, color="r", label='Val MAE')
ax2_loss.set_title("Validation Metrics")
lines3, labels3 = ax2_loss.get_legend_handles_labels()
lines4, labels4 = ax2_acc.get_legend_handles_labels()
ax2_loss.legend(lines3 + lines4, labels3 + labels4, loc='upper center')

fig.tight_layout()
plt.show()