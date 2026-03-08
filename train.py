import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os

# ------------------------------
# Device Setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# Settings
# ------------------------------
DATASET_PATH = "PlantVillage"   # Your PlantVillage folder
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = 224

# ------------------------------
# Image Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Load Dataset
# ------------------------------
full_dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)

# Train / Validation Split (80-20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Total Classes:", num_classes)
print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))

# ------------------------------
# Load Pretrained ResNet50
# ------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ------------------------------
# Loss & Optimizer
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ------------------------------
# Training Loop with tqdm
# ------------------------------
for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print("-" * 40)

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=True)

    for images, labels in progress_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({
            "Loss": f"{running_loss / (total / labels.size(0)):.4f}",
            "Accuracy": f"{100 * correct / total:.2f}%"
        })

    train_accuracy = 100 * correct / total
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    # --------------------------
    # Validation
    # --------------------------
    model.eval()
    correct = 0
    total = 0

    val_bar = tqdm(val_loader, desc="Validation", leave=True)

    with torch.no_grad():
        for images, labels in val_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_bar.set_postfix({
                "Val_Accuracy": f"{100 * correct / total:.2f}%"
            })

    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# ------------------------------
# Save Model (Flask Compatible)
# ------------------------------
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names
}, "plant_model.pth")

print("\nModel saved successfully as plant_model.pth")

# Nirmata -> Neer Chaudhary ^_^