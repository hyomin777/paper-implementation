import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.1
weight_decay = 0.0001
batch_size = 128
epochs = 100