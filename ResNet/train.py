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
    