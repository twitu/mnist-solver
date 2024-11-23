import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(12 * 7 * 7, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )  # Direct mapping to output classes

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.classifier[0].in_features)
        x = self.classifier(x)
        return F.softmax(x, dim=1)


def train_and_test():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enhanced data transformations
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Load Fashion MNIST datasets
    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training loop
    num_epochs = 2

    # Initialize model, loss function, and optimizer
    model = Network().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=1,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy="cos",
    )

    # Count total number of weights
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal number of trainable weights: {total_params}")

    final_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Test accuracy after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%"
        )
        final_accuracy = accuracy

    return final_accuracy


if __name__ == "__main__":
    train_and_test()
