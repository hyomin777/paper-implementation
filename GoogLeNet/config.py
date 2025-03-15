import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 64
lr = 0.01
epochs = 100

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_set = CIFAR100(root='./data', train=True, download=True, transform=transform)
test_set = CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
