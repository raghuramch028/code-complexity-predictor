# ==============================================================================
# 🚀 IMPROVED CODE COMPLEXITY PREDICTOR TRAINING SCRIPT 🚀
# ==============================================================================
# Run this entire script in Google Colab (either pasted into a cell or via script)

# 1. Install dependencies
# !pip install -q transformers datasets torch scikit-learn

import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import shutil

# ------------------------------------------------------------------------------
# ⚙️ CONFIGURATION & HYPERPARAMETERS
# ------------------------------------------------------------------------------
MODEL_NAME = "microsoft/graphcodebert-base"  # 🔥 Upgraded to GraphCodeBERT
MAX_LEN = 512                                # Max token length
BATCH_SIZE = 16                              # Training batch size
EPOCHS = 15                                  # 🔥 Increased from 3 to 15
LEARNING_RATE = 3e-5                         # Optimized initial learning rate
WEIGHT_DECAY = 0.05                          # 🔥 Increased Regularization
PATIENCE = 3                                 # 🔥 Early Stopping patience
SAVE_PATH = "best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

# ------------------------------------------------------------------------------
# 📊 DATA PREPARATION
# ------------------------------------------------------------------------------
print("\n[1/5] Loading Dataset...")
dataset = load_dataset("codeparrot/codecomplex")
df = pd.DataFrame(dataset['train'])

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['complexity'])

# Save Label Encoder for Inference
import joblib
joblib.dump(le, "label_encoder.pkl")

# Train/Test Split (stratified)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Calculate Class Weights to handle imbalance right from the start
class_counts = train_df['label'].value_counts().sort_index().values
total_samples = sum(class_counts)
class_weights = torch.tensor([total_samples / c for c in class_counts], dtype=torch.float).to(device)

print(f"✅ Loaded {len(train_df)} training and {len(test_df)} testing samples.")

# ------------------------------------------------------------------------------
# 🧠 TOKENIZATION & DATASETS
# ------------------------------------------------------------------------------
print(f"\n[2/5] Initializing Tokenizer ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LEN):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.iloc[idx]['src'])
        label = int(self.data.iloc[idx]['label'])

        encoding = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_dataset = CodeDataset(train_df.reset_index(drop=True), tokenizer)
test_dataset = CodeDataset(test_df.reset_index(drop=True), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------------------------------------------------------
# 🏗️ MODEL INITIALIZATION
# ------------------------------------------------------------------------------
print(f"\n[3/5] Loading Model ({MODEL_NAME})...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=7)
model = model.to(device)

# Optimizer with Weight Decay
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),  # 10% warmup
    num_training_steps=total_steps
)

# Loss function with balanced classes
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ------------------------------------------------------------------------------
# 🏃 TRAINING & EVALUATION FUNCTIONS
# ------------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ------------------------------------------------------------------------------
# 🥇 MAIN TRAINING LOOP WITH EARLY STOPPING
# ------------------------------------------------------------------------------
print("\n[4/5] Starting Training Loop...")
best_accuracy = 0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    print(f"\n🔄 Epoch {epoch+1}/{EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
    test_acc = evaluate(model, test_loader, device)

    print(f"📈 Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    # Early Stopping Logic
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"⭐ NEW BEST MODEL SAVED! Accuracy: {best_accuracy*100:.2f}%")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epochs.")

    if epochs_no_improve >= PATIENCE:
        print(f"\n⏹️ EARLY STOPPING TRIGGERED! Test accuracy hasn't improved in {PATIENCE} epochs.")
        break

# ------------------------------------------------------------------------------
# 💾 EXPORTING TO DRIVE
# ------------------------------------------------------------------------------
print("\n[5/5] Finalizing...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    shutil.copy(SAVE_PATH, f"/content/drive/MyDrive/{SAVE_PATH}")
    shutil.copy("label_encoder.pkl", "/content/drive/MyDrive/label_encoder.pkl")
    print("✅ Files successfully backed up to Google Drive!")
except ImportError:
    print("Not running in Colab - skipping Drive export.")
