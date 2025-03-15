import torch

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from config import batch_size


class CustomedCIFAR10(CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

        # (B, H, W, C) → (B, C, H, W)
        self.data:torch.Tensor = torch.tensor(self.data, dtype=torch.float32).permute(0, 3, 1, 2)
        # (1, C, H, W)
        self.per_pixel_mean = self.data.mean(dim=0, keepdim=True)  
        self.data -= self.per_pixel_mean

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


train_transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
])

# test_transform = transforms.Compose([
#     transforms.ToTensor()
# ])

train_dataset = CustomedCIFAR10(root="./data", train=True, transform=train_transform)
test_dataset = CustomedCIFAR10(root="./data", train=False, transform=None)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
