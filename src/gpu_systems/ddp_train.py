"""
Distributed Data Parallel (DDP) Multi-GPU Training Harness
Implements PyTorch NCCL process group initialization, DistributedSampler, 
DDP model wrapping, and multi-GPU rank synchronization.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class DummyDataset(Dataset):
    def __init__(self, size: int = 1000, seq_len: int = 128, vocab_size: int = 10000):
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        self.labels = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def setup_ddp():
    """Initializes NCCL process group using env vars set by torchrun/torch.distributed.run."""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, local_rank, world_size

def cleanup_ddp():
    """Destroys DDP process group."""
    dist.destroy_process_group()

def train_ddp_epoch(model, dataloader, optimizer, criterion, device, rank):
    """Executes single training epoch across DDP ranks."""
    model.train()
    total_loss = 0.0
    for step, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if rank == 0:
        print(f"Rank 0 | Average Epoch Loss: {total_loss / len(dataloader):.4f}")

def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}/{world_size}] Initialized on GPU {local_rank}")

    # Import Transformer model
    try:
        from llm_foundations.Transformer import Transformer
        model_config = {"vocab_size": 10000, "context_length": 128, "num_layers": 4, "d_model": 256, "num_heads": 8, "d_ff": 1024, "rope_theta": 10000.0, "device": local_rank}
        model = Transformer(**model_config).to(device)
    except ImportError:
        # Fallback model
        model = nn.Sequential(nn.Embedding(10000, 256), nn.Linear(256, 10000)).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    dataset = DummyDataset(size=500, seq_len=128, vocab_size=10000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        sampler.set_epoch(epoch)
        train_ddp_epoch(model, dataloader, optimizer, criterion, device, rank)

    cleanup_ddp()

if __name__ == "__main__":
    main()
