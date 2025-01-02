import os
import argparse
import tempfile
import torch
import torch_npu
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def cleanup():
    dist.destroy_process_group()


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
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nproc_per_node', type=int, default=8)
    return parser.parse_args()


def data_process(inputs, labels):
    squeezed_tensor = inputs.squeeze(0).squeeze(0)
    inputs = squeezed_tensor[:, :10]
    labels = labels.repeat(28, 5) * (1 / 140)
    return inputs, labels


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group("hccl", rank=(args.node_rank) * (args.nproc_per_node) + local_rank, world_size=world_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    torch_npu.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    model = ToyModel().to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    step = 0

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_dataloader:
            inputs, labels = data_process(inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()
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

    cleanup()

if __name__ == "__main__":
    main()
