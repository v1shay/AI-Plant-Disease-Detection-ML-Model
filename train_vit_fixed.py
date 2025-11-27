import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
import os

# -------------------------------
# 1. Create saved_model folder
# -------------------------------
os.makedirs("saved_model", exist_ok=True)

# -------------------------------
# 2. Hyperparameters
# -------------------------------
batch_size = 32
num_epochs = 5
learning_rate = 0.0001

# -------------------------------
# 3. Transform (must match API)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------
# 4. Load PlantVillage dataset
# Replace this path with your dataset path
# -------------------------------
train_dir = "plant_dataset/train"
val_dir = "plant_dataset/val"

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# -------------------------------
# 5. Load EfficientNet-B0
# -------------------------------
model = create_model("efficientnet_b0", pretrained=True, num_classes=len(train_dataset.classes))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 6. Training Loop
# -------------------------------
print("Training started...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

print("\nTraining complete!")

# -------------------------------
# 7. Save your model to the EXACT path your API expects
# -------------------------------
model_path = "saved_model/efficientnet_plant_disease.pth"
torch.save(model.state_dict(), model_path)

print(f"\nModel saved to: {model_path}")
