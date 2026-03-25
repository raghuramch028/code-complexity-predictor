!pip install transformers datasets torch scikit-learn

# --- CELL ---

from datasets import load_dataset

dataset = load_dataset("codeparrot/codecomplex")
print(dataset)
print(dataset['train'][0])

# --- CELL ---

import pandas as pd

df = pd.DataFrame(dataset['train'])

# Check complexity labels
print("Complexity classes:")
print(df['complexity'].value_counts())

print("\nLanguages:")
print(df['from'].value_counts())

print("\nTotal samples:", len(df))

# --- CELL ---

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['complexity'])

print("Label mapping:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} → {i}")

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print(f"\nTrain size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# --- CELL ---

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

print("✅ CodeBERT tokenizer loaded!")

# Test it
sample = df['src'][0][:200]
tokens = tokenizer(sample, truncation=True, max_length=512, return_tensors="pt")
print("Sample token shape:", tokens['input_ids'].shape)

# --- CELL ---

import torch
from torch.utils.data import Dataset

class CodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
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

# Create datasets
train_dataset = CodeDataset(train_df.reset_index(drop=True), tokenizer)
test_dataset = CodeDataset(test_df.reset_index(drop=True), tokenizer)

print(f"✅ Train dataset: {len(train_dataset)} samples")
print(f"✅ Test dataset: {len(test_dataset)} samples")

# --- CELL ---

from transformers import AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=7
)

model = model.to(device)
print("✅ CodeBERT model loaded!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- CELL ---

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Scheduler
total_steps = len(train_loader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

print(f"✅ DataLoaders ready!")
print(f"Total training steps: {total_steps}")
print(f"Steps per epoch: {len(train_loader)}")

# --- CELL ---

from tqdm import tqdm

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# Train for 3 epochs
best_accuracy = 0

for epoch in range(3):
    print(f"\n🔄 Epoch {epoch+1}/3")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    test_acc = evaluate(model, test_loader, device)

    print(f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f"✅ Best model saved! Accuracy: {best_accuracy*100:.2f}%")

# --- CELL ---

# Train 2 more epochs
for epoch in range(2):
    print(f"\n🔄 Epoch {epoch+4}/5")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    test_acc = evaluate(model, test_loader, device)

    print(f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f"✅ Best model saved! Accuracy: {best_accuracy*100:.2f}%")

# --- CELL ---

from google.colab import drive
drive.mount('/content/drive')

# --- CELL ---

import shutil

# Copy files to Google Drive
shutil.copy("best_model.pt", "/content/drive/MyDrive/best_model.pt")
shutil.copy("label_encoder.pkl", "/content/drive/MyDrive/label_encoder.pkl")

print("✅ Files saved to Google Drive!")

# --- CELL ---

# Test the model directly in Colab
test_codes = [
    "public int findMax(int[] arr) { int max = arr[0]; for (int i = 1; i < arr.length; i++) { if (arr[i] > max) max = arr[i]; } return max; }",
    "return arr[0];",
    "for(int i=0;i<n;i++) for(int j=0;j<n;j++) sum+=arr[i][j];",
]

for code in test_codes:
    inputs = tokenizer(code, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()

    print(f"Code: {code[:50]}...")
    print(f"Predicted: {le.inverse_transform([pred])[0]}\n")

# --- CELL ---

import torch.nn as nn

# Count class frequencies
class_counts = df['label'].value_counts().sort_index().values
total = sum(class_counts)
class_weights = torch.tensor([total/c for c in class_counts], dtype=torch.float).to(device)

print("Class weights:", class_weights)

# New training loop with weighted loss
def train_epoch_weighted(model, loader, optimizer, scheduler, device, weights):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(weight=weights)

    for batch in tqdm(loader, desc="Training"):
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

# Retrain with weights
optimizer3 = AdamW(model.parameters(), lr=5e-6)
scheduler3 = get_linear_schedule_with_warmup(optimizer3, num_warmup_steps=30, num_training_steps=len(train_loader)*3)

for epoch in range(3):
    print(f"\n🔄 Epoch {epoch+1}/3")
    train_loss, train_acc = train_epoch_weighted(model, train_loader, optimizer3, scheduler3, device, class_weights)
    test_acc = evaluate(model, test_loader, device)
    print(f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), "best_model.pt")
        print(f"✅ Best model saved! Accuracy: {best_accuracy*100:.2f}%")

# --- CELL ---

import shutil
shutil.copy("best_model.pt", "/content/drive/MyDrive/best_model.pt")
print("✅ Saved to Google Drive!")