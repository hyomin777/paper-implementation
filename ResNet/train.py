import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from tqdm import tqdm

from model import ResNet
from dataset import train_loader, test_loader
from config import device, lr, weight_decay, epochs


def train_main(load_state=False, weights_path=None):
    model = ResNet(num_classes=10).to(device)
    if load_state and weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
        print("Loaded saved model weights.")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32000 // len(train_loader), 48000 // len(train_loader)], gamma=0.1)

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        train_total_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in tqdm(train_loader):
            x:Tensor = x.to(device)
            y:Tensor = y.to(device)

            output:Tensor = model(x)
            loss = F.cross_entropy(output, y).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_total_loss += loss.item()
            _, predicted = output.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        scheduler.step()
        train_avg_loss = train_total_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        print(f"Epoch: {epoch}, Loss: {train_avg_loss:.3f}, Accuracy: {train_accuracy:.2f}")

        if epoch % 10 == 0:
            weights_name = f"model_weights_{epoch}.pth"
            torch.save(model.state_dict(), weights_name)
            print(f"Model weights:{weights_name} saved")

        with torch.no_grad():
            model.eval()
            test_total_loss = 0.0
            test_correct = 0
            test_total = 0
   
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                output:Tensor = model(x)
                
                loss = F.cross_entropy(output, y)
                test_total_loss += loss.item()

                _, predicted = output.max(1)
                test_total += y.size(0)
                test_correct += predicted.eq(y).sum().item()

            avg_loss = test_total_loss / len(test_loader)
            accuracy = 100.0 * test_correct / test_total
            print(f"Test Loss: {avg_loss:.3f}, Test Accuracy: {accuracy:.2f}%")



if __name__ == "__main__":
    train_main(load_state=True, weights_path="model_weights_10.pth")