import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from loss import ContentLoss, StyleLoss
from model import StyleTransfer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def pre_processing(image:Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    image_tensor:torch.Tensor = transform(image)
    return image_tensor.unsqueeze(0)

def post_processing(tensor:torch.Tensor) -> Image.Image:
    # (1, C, H, W)
    image:np.ndarray = tensor.to(device).detach().numpy()
    # (C, H, W)
    image = image.squeeze()
    # (H, W, C)
    image = image.transpose(1, 2, 0)
    image = image * std + mean
    # clip
    image = image.clip(0.1)*255
    image = image.astype(np.unit8)
    return Image.fromarray(image)


def train_main():
    content_image = Image.open('./data/content.jpg')
    content_image = pre_processing(content_image)

    style_image = Image.open('./data/style.jpg')
    style_image = pre_processing(style_image)

    style_transfer = StyleTransfer().eval()

    content_loss = ContentLoss()
    style_loss = StyleLoss()

    alpha = 1
    beta = 1
    lr = 0.01

    style_transfer = style_transfer.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)
        
    x = torch.randn(1, 3, 512, 512).to(device)
    x.requires_grad_(True)

    optimizer = optim.Adam([x], lr=lr)

    epochs = 1000
    for epoch in range(epochs):
        x_content_list = style_transfer(x, 'content')
        y_content_list = style_transfer(content_image, 'content')

        x_style_list = style_transfer(x, 'style')
        y_style_list = style_transfer(style_image, 'style')

        loss_c = 0
        loss_s = 0
        loss_total = 0

        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c += content_loss(x_content, y_content)
        loss_c = alpha * loss_c

        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_s += style_loss(x_style, y_style)
        loss_s = beta * loss_s
        
        loss_total = loss_c + loss_s

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"loss_c: {loss_c.cpu()}")
            print(f"loss_s: {loss_s.cpu()}")
            print(f"loss_total: {loss_total.cpu()}")

            gen_img:Image.Image = post_processing(x)
            os.path.join('./result.jpg', f'{epoch}.jpg')
            gen_img.save()


if __name__ == "__main__":
    train_main()
