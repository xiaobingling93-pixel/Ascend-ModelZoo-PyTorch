import argparse

import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch_npu.npu import amp


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def parse_args():
    parser = argparse.ArgumentParser(description="command line arguments")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    return parser.parse_args()


def data_process(inputs, labels):
    squeezed_tensor = inputs.squeeze(0).squeeze(0)
    inputs = squeezed_tensor[:, :10]
    labels = labels.repeat(28, 5) * (1 / 140)
    return inputs, labels


def main():
    args = parse_args()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("npu")

    model = ToyModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = amp.GradScaler()
    step = 0

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = data_process(inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)

            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels).to(device)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"step = {step}")
            step += 1
        checkpoint_path = "checkpoint.pth.tar"
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, checkpoint_path)
    
    save_pt_path = "state_dict_model.pt"
    torch.save(model.state_dict(), save_pt_path)

if __name__ == "__main__":
    main()
