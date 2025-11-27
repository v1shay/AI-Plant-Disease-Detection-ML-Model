import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import kagglehub
import timm
from torch import nn, optim

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

# -----------------------------
# 1. DOWNLOAD KAGGLE DATASET
# -----------------------------
print("Downloading dataset from Kaggle...")
data_path = kagglehub.dataset_download("karagwaanntreasure/plant-disease-detection")
print("Dataset downloaded to:", data_path)

dataset_dir = os.path.join(data_path, "Dataset")

# -----------------------------
# 2. IMAGE TRANSFORMS
# -----------------------------
image_size = 224

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 3. LOAD DATASET
# -----------------------------
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
original_classes = dataset.classes

print("Applying FAST MODE: Using only 2000 images...")
indices = list(range(min(2000, len(dataset))))
dataset = torch.utils.data.Subset(dataset, indices)

dataset.classes = original_classes

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0)

num_classes = len(dataset.classes)
print("Detected classes:", num_classes)
print(dataset.classes)

# -----------------------------
# 4. LOAD MODEL (EfficientNet)
# -----------------------------
model = timm.create_model(
    'efficientnet_b0.ra_in1k',
    pretrained=True,
    num_classes=num_classes,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# -----------------------------
# 5. LOSS + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# -----------------------------
# 6. TRAINING LOOP
# -----------------------------
epochs = 5

for epoch in range(epochs):
    print(f"\nStarting Epoch {epoch + 1}/{epochs}")

    # Reset training loss accumulator
    train_loss = 0.0

    # Training loop
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")

    # -----------------------------
    # VALIDATION LOOP
    # -----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    model.train()

# -----------------------------
# 7. SAVE MODEL
# -----------------------------
os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), "saved_model/efficientnet_plant_disease.pth")

print("\nTraining complete!")
print("Model saved to saved_model/efficientnet_plant_disease.pth")
