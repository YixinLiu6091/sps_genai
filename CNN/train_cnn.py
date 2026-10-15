
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from CNN.cnn_model import CNNModel


# 1. Set up paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "cnn_model.pth"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# 2. Select device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# 3. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# 4. Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


# 5. Initialize CNN model
model = CNNModel().to(device)


# 6. Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)


# 7. Train the CNN
EPOCHS = 10

datalogs = []

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    train_loader_with_progress = tqdm(
        iterable=train_loader,
        ncols=120,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for batch_number, (images, labels) in enumerate(
        train_loader_with_progress
    ):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Track training performance
        running_correct += (
            predicted == labels
        ).sum().item()

        running_total += labels.size(0)
        running_loss += loss.item()

        # Update progress bar
        if batch_number % 100 == 99:

            avg_accuracy = (
                running_correct / running_total
            )

            avg_loss = (
                running_loss / (batch_number + 1)
            )

            train_loader_with_progress.set_postfix({
                "avg accuracy": f"{avg_accuracy:.3f}",
                "avg loss": f"{avg_loss:.4f}"
            })

            datalogs.append({
                "epoch": epoch + (
                    batch_number / len(train_loader)
                ),
                "train_loss": avg_loss,
                "train_accuracy": avg_accuracy
            })

    # Epoch summary
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = running_correct / running_total

    datalogs.append({
        "epoch": epoch + 1,
        "train_loss": epoch_loss,
        "train_accuracy": epoch_accuracy
    })

    print(
        f"Epoch [{epoch+1}/{EPOCHS}], "
        f"Loss: {epoch_loss:.4f}, "
        f"Accuracy: {epoch_accuracy*100:.2f}%"
    )

print("Finished Training")


# 8. Evaluate the model
model.eval()

correct = 0
total = 0

with torch.no_grad():

    test_loader_with_progress = tqdm(
        test_loader,
        desc="Testing",
        ncols=120
    )

    for images, labels in test_loader_with_progress:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")


# 9. Save trained model
torch.save(
    model.cpu().state_dict(),
    MODEL_PATH
)

print(f"Model saved to: {MODEL_PATH}")
