import argparse

import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch_npu.npu import amp


class ToyModel1(nn.Module):
    def __init__(self):
        super(ToyModel1, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class ToyModel2(nn.Module):
    def __init__(self):
        super(ToyModel2, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu1 = nn.ReLU()
        self.net2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.net3 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net3(self.relu2(self.net2(self.relu1(self.net1(x)))))


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

    model1 = ToyModel1().to(device)
    model2 = ToyModel2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.learning_rate)
    scaler = amp.GradScaler()
    step = 0

    for epoch in range(args.epochs):
        model1.train()
        model2.train()
        for inputs, labels in train_dataloader:
            inputs, labels = data_process(inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)

            with amp.autocast():
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)
                loss1 = criterion(outputs1, labels).to(device)
                loss2 = criterion(outputs2, labels).to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            scaler.scale(loss1).backward(retain_graph=True)
            scaler.scale(loss2).backward()
            scaler.step(optimizer1)
            scaler.step(optimizer2)
            scaler.update()
            print(f"step = {step}")
            step += 1
        checkpoint_path = "checkpoint.pth.tar"
        torch.save({
            'epoch': epoch,
            'loss': loss1,
            'state_dict': model1.state_dict(),
            'optimizer': optimizer1.state_dict(),
        }, checkpoint_path)

    save_pt_path = "state_dict_model.pt"
    torch.save(model1.state_dict(), save_pt_path)


if __name__ == "__main__":
    main()
