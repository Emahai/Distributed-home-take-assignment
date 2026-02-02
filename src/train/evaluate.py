import torch
from src.common.correctness import accuracy

@torch.no_grad()
def evaluate(model, dataloader, device="cpu"):
    model.eval()
    total_acc = 0.0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_acc += accuracy(logits, y) * x.size(0)
        total += x.size(0)
    return total_acc / max(total, 1)
