import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch_npu
from torchvision import datasets, transforms

DATA_DIR = "./data"


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


def data_process(inputs, labels):
    inputs = inputs.view(-1, 784)
    labels = labels.view(-1)
    return inputs, labels


def get_train_args():
    parser = argparse.ArgumentParser(description="command line arguments")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    return args


def train(args):
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank_idx = int(os.environ["RANK"])
    local_rank_idx = args.local_rank
    
    devices_per_node = torch.npu.device_count()

    dist.init_process_group("hccl", rank=global_rank_idx, world_size=world_size, timeout=timedelta(minutes=30))
    torch_npu.npu.set_device(local_rank_idx)

    device_id = f"npu:{local_rank_idx}"

    model = ToyModel().to(device_id)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=devices_per_node, rank=local_rank_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    criterion = nn.CrossEntropyLoss().to(device_id)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    model = DDP(model)

    for epoch in range(args.epochs):
        if local_rank_idx == 0:
            print(f"\nCurrent epoch: {epoch}")
            
        train_sampler.set_epoch(epoch)

        model.train()

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = data_process(inputs, labels)
            inputs, labels = inputs.to(device_id), labels.to(device_id)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_rank_idx == 0:
                print(f"Current step: {i}, loss: {loss.item()}")

    cleanup()


def main():
    args = get_train_args()
    train(args)


if __name__ == "__main__":
    main()