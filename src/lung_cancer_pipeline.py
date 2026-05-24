import warnings
warnings.filterwarnings("ignore")
import os, sys, json, shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")

CLASS_NAMES = ["adenocarcinoma", "large_cell_carcinoma", "normal", "squamous_cell_carcinoma"]
CLASS_LABELS = {0: "Adenocarcinoma", 1: "Large Cell Carcinoma", 2: "Normal", 3: "Squamous Cell Carcinoma"}
TUMOR_GRADES = {
    "adenocarcinoma": {"stage": "II-IV", "description": "Glandular tissue malignancy", "risk": "High"},
    "large_cell_carcinoma": {"stage": "II-III", "description": "Undifferentiated large cells", "risk": "High"},
    "squamous_cell_carcinoma": {"stage": "I-III", "description": "Squamous epithelial malignancy", "risk": "Moderate-High"},
}

print("=" * 60)
print("  Lung Cancer Detection & Classification Pipeline")
print("=" * 60)

print("\n[Step 1/5] Loading dataset...")
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
print(f"  Train: {len(train_dataset)} images ({len(train_dataset.classes)} classes)")
print(f"  Test:  {len(test_dataset)} images ({len(test_dataset.classes)} classes)")
print(f"  Classes: {train_dataset.classes}")
class_counts = pd.Series(train_dataset.targets).value_counts().sort_index()
for idx, count in class_counts.items():
    print(f"    {train_dataset.classes[idx]}: {count}")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

print("\n[Step 2/5] Loading pre-trained ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, len(CLASS_NAMES)),
)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  ResNet18 loaded: {total_params:,} total, {trainable_params:,} trainable")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

print("\n[Step 3/5] Training...")
num_epochs = 8
train_losses, train_accs = [], []
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    scheduler.step(epoch_loss)
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))
    if (epoch + 1) % 4 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

print(f"\n  Training complete! Best accuracy: {best_acc:.4f}")

print("\n[Step 4/5] Evaluating...")
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth"), map_location=device))
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"\n  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\n  Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0, digits=4))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix - Lung Cancer Classification", fontsize=14, fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

per_class_acc = {}
for i, name in enumerate(CLASS_NAMES):
    mask = np.array(all_labels) == i
    if mask.sum() > 0:
        acc = (np.array(all_preds)[mask] == np.array(all_labels)[mask]).mean()
        per_class_acc[name] = float(acc)
print(f"\n  Per-class Accuracy:")
for cls, acc in per_class_acc.items():
    print(f"    {cls}: {acc:.4f} ({acc*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, len(train_losses)+1), train_losses, "b-o", markersize=4)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.3)
axes[1].plot(range(1, len(train_accs)+1), train_accs, "g-o", markersize=4)
axes[1].set_title("Training Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(alpha=0.3)
plt.suptitle("ResNet18 Training History - Lung Cancer CT", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "training_history.png"), dpi=150)
plt.close()

print("\n[Step 5/5] Saving results...")
results = {
    "model": "ResNet18 (ImageNet pretrained, transfer learning)",
    "dataset": "Lung Cancer CT Scans (Mahadih534/Chest_CT-Scan_images-Dataset)",
    "dataset_source": "IQ-OTH/NCCD Lung Cancer Dataset",
    "num_classes": len(CLASS_NAMES),
    "classes": CLASS_NAMES,
    "train_samples": len(train_dataset),
    "test_samples": len(test_dataset),
    "test_accuracy": float(accuracy),
    "per_class_accuracy": per_class_acc,
    "parameters": {"total": total_params, "trainable": trainable_params},
    "timestamp": datetime.now().isoformat(),
}
with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("  RESULTS SUMMARY")
print("="*60)
print(f"  Test Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Best epoch:         {train_accs.index(max(train_accs))+1}")
print(f"  Classes:            {' | '.join(CLASS_NAMES)}")
print(f"  Cancer Types:       Adenocarcinoma | Large Cell | Squamous Cell")
print(f"\n  Results saved to:   {RESULTS_DIR}")
print(f"  Model saved to:     {MODELS_DIR}/best_model.pth")
