import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from tqdm import tqdm

from model import GoogLeNet
from loss import GoogLeNetLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu' 
batch_size = 64
lr = 0.01
epochs = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_set = CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

def train_main():
    model = GoogLeNet(num_classes=100).to(device)
    model.load_state_dict(torch.load("model_weights.pth"))
    criterion = GoogLeNetLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0

        for x, y in train_loader:
            x:Tensor = x.to(device)
            y:Tensor = y.to(device)

            y_hat, aux1_y, aux2_y = model(x)
            loss:Tensor = criterion(y_hat, y, aux1_y, aux2_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch: {epoch}, Loss: {total_loss}")
        torch.save(model.state_dict(), "model_weights.pth")


if __name__ == "__main__":
    train_main()