import os
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Normalize data
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.1307,), (.3081,))
])

# Train and test data
train_data = datasets.MNIST(root='cnn_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='cnn_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    img = images[i].squeeze()
    ax.imshow(img)
    ax.axis("off")
plt.tight_layout()
plt.show()

# Define model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),      # 28x28
            nn.Conv2d(32, 64, 3, padding = 1), nn.ReLU(),   
            nn.MaxPool2d(2),                                # 14x14
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)                                 # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
# Instantiate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

# Train the model
epochs = 10

best_acc, best_state = 0.0, None
train_losses, val_losses, val_accs = [], [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_data)

    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item(): .4f}")

# Validate
    model.eval()
    val_loss, correct, total = 0.0, 0, 0 
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    val_loss /= len(test_data)
    val_acc = 100 * correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_state = {k:v.cpu() for k, v in model.state_dict().items()}

    print(f"Epoch: {epoch+1} / {epochs} | train_loss {train_loss: .4f} | val_loss {val_loss: .4f} | val_acc {val_acc: .2f}%")

# Save model
torch.save(best_state, "models/mnist_cnn_best.pth")

# Learning curve and results
plt.figure()
plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('results/loss_curve.png', dpi=150)

plt.figure()
plt.plot(val_accs)
plt.xlabel('epoch')
plt.ylabel('val acc %')
plt.savefig('results/val_acc.png', dpi=150)

# Confusion matrix
model.load_state_dict(torch.load("models/mnist_cnn_best.pth"))
model.eval()
all_y, all_p = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu()
        all_p.append(pred)
        all_y.append(yb)

all_p = torch.cat(all_p).numpy()
all_y = torch.cat(all_y).numpy()

cm= confusion_matrix(all_y, all_p)
print("Confusion matrix: \n", cm)