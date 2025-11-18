!pip install pytorch-lightning transformers
!pip install captum
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
import transformers
from captum.attr import LayerIntegratedGradients
import seaborn as sns
from wordcloud import WordCloud

PATH_TO_TRAIN_DATA = 'train.csv'
df = pd.read_csv(PATH_TO_TRAIN_DATA)
df['review'] = df['positive'] + ' ' + df['negative']

df_train, df_test = train_test_split(df, random_state=1412, train_size=0.75)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

class ReviewDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_test, tokenizer, batch_size=16, max_len=128):
        super().__init__()
        self.df_train = df_train
        self.df_test = df_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    def setup(self, stage=None):
        self.train_dataset = ReviewDataset(
            texts=self.df_train['review'].values,
            labels=self.df_train['score'].values,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

        self.val_dataset = ReviewDataset(
            texts=self.df_test['review'].values,
            labels=self.df_test['score'].values,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

class MAEVisualizationCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_mae = []
        self.val_mae = []
        self.epochs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get metrics from the current epoch
        train_mae = trainer.callback_metrics.get('train_mae_epoch', None)
        val_mae = trainer.callback_metrics.get('val_mae', None)

        if train_mae is not None and val_mae is not None:
            self.epochs.append(trainer.current_epoch)
            self.train_mae.append(train_mae.item())
            self.val_mae.append(val_mae.item())

            # Plot the MAE curve
            self.plot_mae()

    def plot_mae(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_mae, marker='o', label='Training MAE', linewidth=2)
        plt.plot(self.epochs, self.val_mae, marker='s', label='Validation MAE', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.title('MAE over Training Epochs, Transformer', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('mae_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

class TransformerRegressor(pl.LightningModule):
    def __init__(self, model_name="distilbert-base-uncased", learning_rate=5e-5, warmup_proportion=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(
            self.transformer.config.hidden_size,
            1
        )
        self.criterion = nn.L1Loss()

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        cls_token_output = self.dropout(cls_token_output)
        prediction = self.linear(cls_token_output)
        return prediction

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        preds = self(input_ids, attention_mask).squeeze()
        loss = self.criterion(preds, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae_epoch', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        preds = self(input_ids, attention_mask).squeeze()
        loss = self.criterion(preds, labels)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        num_training_steps = self.trainer.max_steps
        num_warmup_steps = int(num_training_steps * self.hparams.warmup_proportion)

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]

def analyze_word_importance(model, tokenizer, sample_texts, device='cuda' if torch.cuda.is_available() else 'cpu'):

    model.eval()
    model.to(device)

    word_importance_scores = {}

    for text in sample_texts:
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Get embeddings and detach to create a leaf variable
        with torch.no_grad():
            embeddings = model.transformer.embeddings.word_embeddings(input_ids)

        # Create a new leaf tensor that requires grad
        embeddings = embeddings.detach().clone()
        embeddings.requires_grad = True

        # Forward pass with embeddings
        outputs = model.transformer(inputs_embeds=embeddings, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = model.dropout(cls_output)
        output = model.linear(cls_output)

        # Calculate gradients
        model.zero_grad()
        output.backward()

        # Get gradient magnitude for each token
        gradients = embeddings.grad.abs().sum(dim=-1).squeeze().cpu().detach().numpy()

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())

        # Aggregate importance scores
        for token, score in zip(tokens, gradients):
            if token not in ['[PAD]', '[CLS]', '[SEP]']:
                if token in word_importance_scores:
                    word_importance_scores[token].append(score)
                else:
                    word_importance_scores[token] = [score]

    avg_importance = {token: np.mean(scores) for token, scores in word_importance_scores.items()}

    return avg_importance

def plot_word_importance(word_scores, top_n=20):

    # Sort by importance
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    words, scores = zip(*sorted_words)

    # 1. Bar chart
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(words)), scores, color='steelblue')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Words/Tokens', fontsize=12)
    plt.title(f'Top {top_n} Most Important Words for Prediction', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('word_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Word Cloud
    top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:100]
    word_freq = {word: score for word, score in top_words}

    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        colormap='viridis',
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_freq)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Importance Cloud (Size = Importance Score)',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout(pad=0)
    plt.savefig('word_importance_cloud.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nTop {top_n} Most Important Words:")
    for i, (word, score) in enumerate(sorted_words, 1):
        print(f"{i:2d}. {word:20s} - Score: {score:.4f}")

if __name__ == "__main__":
    # Set hyperparameters
    MODEL_NAME = "distilbert-base-uncased"
    BATCH_SIZE = 64
    MAX_LEN = 128
    LEARNING_RATE = 5e-5
    WARMUP_PROPORTION = 0.15
    N_EPOCHS = 5

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. Initialize DataModule
    data_module = ReviewDataModule(
        df_train=df_train,
        df_test=df_test,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )

    # 3. Initialize Model
    model = TransformerRegressor(
        MODEL_NAME,
        learning_rate=LEARNING_RATE,
        warmup_proportion=WARMUP_PROPORTION
    )

    # 4. Define Callbacks
    early_stopping = EarlyStopping(
        monitor='val_mae',
        mode='min',
        patience=2
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='best-model',
        monitor='val_mae',
        mode='min'
    )

    # MAE visualization callback
    mae_viz_callback = MAEVisualizationCallback()

    # 5. Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[early_stopping, checkpoint_callback, mae_viz_callback],
        logger=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1
    )

    print("--- Starting Model Training ---")
    trainer.fit(model, data_module)
    print("--- Training Finished ---")

    # 6. Validate the best model
    print("--- Validating Best Model ---")
    val_results = trainer.validate(datamodule=data_module, ckpt_path='best')
    print(val_results)

    print("\n--- Analyzing Word Importance ---")

    best_model = TransformerRegressor.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model_name=MODEL_NAME
    )

    sample_texts = df_test['review'].head(100).tolist()

    word_scores = analyze_word_importance(best_model, tokenizer, sample_texts)

    plot_word_importance(word_scores, top_n=20)

    print("\n--- Analysis Complete ---")
    print("Visualizations saved:")
    print("  - mae_curve.png: Training and validation MAE over epochs")
    print("  - word_importance_bar.png: Top 20 most important words (bar chart)")
    print("  - word_importance_cloud.png: Word cloud where size = importance")