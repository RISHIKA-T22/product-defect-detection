import os
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# =========================
# 1. Dataset Preparation
# =========================

data_dir = "dataset"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

base_dataset = datasets.ImageFolder(data_dir)

print("Classes:", base_dataset.classes)

# Get correct labels from folder names
defective_label = base_dataset.class_to_idx["def_front"]
normal_label = base_dataset.class_to_idx["ok_front"]

defective_indices = [
    i for i, (_, label) in enumerate(base_dataset.samples)
    if label == defective_label
]

normal_indices = [
    i for i, (_, label) in enumerate(base_dataset.samples)
    if label == normal_label
]

# Select 50 images from each class
random.seed(42)

selected_indices = (
    random.sample(defective_indices, 50) +
    random.sample(normal_indices, 50)
)

random.shuffle(selected_indices)

# 80% training, 20% validation
train_indices = selected_indices[:80]
val_indices = selected_indices[80:]

train_dataset = datasets.ImageFolder(
    data_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    data_dir,
    transform=val_transform
)

train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

print("Total selected images:", 100)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# 2. Transfer Learning
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Load pretrained ResNet18
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final ResNet block
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    2
)
print("Frozen layers: ResNet18 except final block")
print("Trainable layers: ResNet18 layer4 + final fully connected layer")

model = model.to(device)


# =========================
# 3. Training
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

epochs = 8

for epoch in range(epochs):

    # Training
    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (
            predicted == labels
        ).sum().item()

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total

    # Validation
    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


# =========================
# 4. Evaluation
# =========================

model.eval()

all_labels = []
all_predictions = []
error_images = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

        # Store incorrect predictions
        for i in range(len(labels)):

            if predictions[i] != labels[i]:

                error_images.append((
                    images[i].cpu(),
                    labels[i].item(),
                    predictions[i].item()
                ))


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=defective_label,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=defective_label,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    pos_label=defective_label,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n===== Final Results =====")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1-Score :", round(f1, 4))

print("\nConfusion Matrix:")
print(cm)


# =========================
# 5. Confusion Matrix
# =========================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(5, 4))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    base_dataset.classes
)

plt.yticks(
    [0, 1],
    base_dataset.classes
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png"
)

plt.close()


# =========================
# 6. Error Analysis
# =========================

# Save up to 3 incorrectly classified images

for i, (image, actual, predicted) in enumerate(
    error_images[:3]
):

    # Undo ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    )

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    )

    image = image * std[:, None, None] + mean[:, None, None]

    image = torch.clamp(image, 0, 1)

    image = image.permute(1, 2, 0)

    plt.figure(figsize=(4, 4))

    plt.imshow(image)

    plt.title(
        f"Actual: {base_dataset.classes[actual]}\n"
        f"Predicted: {base_dataset.classes[predicted]}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        f"results/error_{i + 1}.png"
    )

    plt.close()


print(
    "\nIncorrect predictions found:",
    len(error_images)
)

print("Results saved in the results folder.")

print(
    "\nIncorrect predictions found:",
    len(error_images)
)

print("Results saved in the results folder.")


# Save trained model
os.makedirs("model", exist_ok=True)

torch.save(
    model.state_dict(),
    "model/trained_model.pth"
)

print("Trained model saved successfully.")