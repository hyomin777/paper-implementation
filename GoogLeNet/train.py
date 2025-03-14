import torch
import torch.optim as optim
from torch import Tensor

from tqdm import tqdm
from model import GoogLeNet
from loss import GoogLeNetLoss
from config import device, lr, epochs, train_loader


def train_main():
    model = GoogLeNet(num_classes=100).to(device)
    model.load_state_dict(torch.load("model_weights.pth"))
    model.train()

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