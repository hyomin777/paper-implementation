import torch
import torch.optim as optim
from torch import Tensor

from tqdm import tqdm
from model import GoogLeNet
from loss import GoogLeNetLoss
from config import device, lr, epochs, train_loader, test_loader


def train_main(load_state=False, weights_path=None):
    model = GoogLeNet(num_classes=100).to(device)
    if load_state and weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    criterion = GoogLeNetLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        model.train()
        train_total_loss = 0.0

        for x, y in train_loader:
            x:Tensor = x.to(device)
            y:Tensor = y.to(device)

            output, aux1_output, aux2_output = model(x)
            loss:Tensor = criterion(output, y, aux1_output, aux2_output)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            
        train_avg_loss = train_total_loss / len(train_loader)
        print(f"Epoch: {epoch}, Loss: {train_avg_loss:.3f}")
        torch.save(model.state_dict(), f"model_weights_{epoch}.pth")

        with torch.no_grad():
            model.eval()
            test_total_loss = 0.0
            correct = 0
            total = 0
   
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                outputs, _, _ = model(x)
                
                loss = torch.nn.functional.cross_entropy(outputs, y)
                test_total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            avg_loss = test_total_loss / len(test_loader)
            accuracy = 100.0 * correct / total
            print(f"Test Loss: {avg_loss:.3f}, Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    train_main(load_state=True, weights_path="model_weights.pth")