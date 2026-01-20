import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from dataset import MultimodalDataset
from model import MultimodalClassifier
from utils import split_train_val
from tqdm import tqdm

# Configuration
DATA_DIR = "data"
TRAIN_FILE = "train.txt"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(fusion_type='concat', save_path=None):
    if save_path is None:
        save_path = f"best_model_{fusion_type}.pth"

    _, preprocess = clip.load("ViT-B/32", device=DEVICE)
    train_data, val_data = split_train_val(TRAIN_FILE)

    train_dataset = MultimodalDataset(train_data, DATA_DIR, preprocess, modality=fusion_type)
    val_dataset = MultimodalDataset(val_data, DATA_DIR, preprocess, modality=fusion_type)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = MultimodalClassifier(fusion_type=fusion_type).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR)

    best_acc = 0.0

    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(EPOCHS), desc=f"[{fusion_type}] Training", position=0, leave=True)
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Inner progress bar for training batches
        train_pbar = tqdm(train_loader, desc="Training", position=1, leave=False, disable=False)
        for batch in train_pbar:
            image = batch['image'].to(DEVICE, non_blocking=True)
            text = batch['text'].to(DEVICE, non_blocking=True)
            label = batch['label'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            output = model(image, text)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update inner progress bar with current loss
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                image = batch['image'].to(DEVICE, non_blocking=True)
                text = batch['text'].to(DEVICE, non_blocking=True)
                label = batch['label'].to(DEVICE, non_blocking=True)
                output = model(image, text)
                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        val_acc = correct / total

        # Update outer progress bar with epoch info
        epoch_pbar.set_postfix({
            "Train Loss": f"{avg_loss:.4f}",
            "Val Acc": f"{val_acc:.4f}"
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n[{fusion_type}] ✅ Training finished. Best Val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train_model('concat')
    train_model('text_only')
    train_model('image_only')