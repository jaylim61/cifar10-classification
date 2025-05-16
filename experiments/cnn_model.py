import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

batch_size = 32
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_dropout = nn.Dropout2d(p=0.2)
        self.fc_dropout = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(in_features=128*4*4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [3, 32, 32] -> [32, 16, 16]
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [32, 16, 16] -> [64, 8, 8]
        x = self.conv_dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [64, 8, 8] -> [128, 4, 4]
        x = self.conv_dropout(x)

        x = torch.flatten(x, 1)                         # [128, 4, 4] -> [2048]
        x = F.relu(self.fc1(x))                         # [2048] -> [256]
        x = self.fc_dropout(x)
        x = self.fc2(x)                                 # [256] -> [10]

        return x

def load_data(train_data, test_data, batch_size):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader

def train(net, train_loader, loss_function, optimizer, epochs):
    net.train()

    for epoch in range(epochs):
        print(f'Training epoch {epoch+1}...')

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Loss: {running_loss / len(train_loader):.4f}')

def test(net, test_loader):
    net.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f'Accuracy: {accuracy}%')
