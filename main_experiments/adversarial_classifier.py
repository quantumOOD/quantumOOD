"""Surrogate classifier used as the attack target for crafting adversarial examples."""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MNIST_NORM = {"mnist": (0.1307, 0.3081), "fashion_mnist": (0.2860, 0.3530)}


class LeNet5_16x16(nn.Module):
    """LeNet-5 family, sized for 16x16 grayscale input."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def features(self, x):
        h = self.pool(torch.relu(self.conv1(x)))  # 16 -> 8
        h = self.pool(torch.relu(self.conv2(h)))  # 8 -> 4
        h = h.reshape(h.size(0), -1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return h  # (N, 84)

    def forward(self, x):
        return self.fc3(self.features(x))


def preprocess_for_classifier(imgs, dataset):
    """imgs: (N,1,16,16) float32 in [0,1], already at the target resolution
    (no resize). Applies the standard published
    per-dataset MNIST/Fashion-MNIST normalization."""
    mean, std = MNIST_NORM[dataset]
    return (imgs - mean) / std


def pretrain_classifier_16x16(dataset, data_dir, img_size=16, epochs=10, lr=1e-3, batch_size=128,
                               checkpoint_path=None, force_retrain=False, seed=0, device=None):
    """Trains LeNet5_16x16 via ordinary 10-class classification on the FULL
    training split (all classes) at native 16x16 resolution -- the
    surrogate model FGSM/PGD/SPSA attack and Salt-and-Pepper's success
    check query. Cached to checkpoint_path (skips retraining if it already
    exists, unless force_retrain=True)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path and os.path.exists(checkpoint_path) and not force_retrain:
        print(f"Loading cached surrogate classifier from '{checkpoint_path}'")
        model = LeNet5_16x16().to(device)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    torch.manual_seed(seed)
    if dataset == "mnist":
        train_set = datasets.MNIST(root=data_dir, train=True, download=True,
                                    transform=transforms.Compose([transforms.Resize((img_size, img_size)),
                                                                   transforms.ToTensor()]))
    else:
        train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                           transform=transforms.Compose([transforms.Resize((img_size, img_size)),
                                                                          transforms.ToTensor()]))

    model = LeNet5_16x16().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = preprocess_for_classifier(x, dataset)
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)
        print(f"surrogate classifier ({dataset}) epoch {ep + 1}/{epochs}: loss={total_loss / n:.4f}, acc={correct / n:.4f}")
    model.eval()

    if checkpoint_path:
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        print(f"Surrogate classifier saved to '{checkpoint_path}'")
    return model
