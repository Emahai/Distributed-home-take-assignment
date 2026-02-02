import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.common.utils import set_seed, ensure_dir
from src.common.timers import Timer
from src.common.logging import write_csv
from src.common.mpitools import get_comm, rank_world, broadcast_model_params, sync_gradients_avg
from src.data.dataset import get_datasets
from src.data.sampler import ShardSampler
from src.models.cnn_full import SimpleCNN
from src.train.evaluate import evaluate
from src.common.correctness import accuracy

def _effective_batch(cfg, world):
    bs = cfg["train"]["batch_size"]
    policy = cfg.get("strategy", {}).get("batch_policy", "global_fixed")
    if policy == "global_fixed":
        # global fixed => per-rank batch size smaller
        return max(1, bs // world)
    if policy == "per_rank_fixed":
        # per-rank fixed => global batch grows
        return bs
    raise ValueError(f"Unknown batch_policy: {policy}")

def run_dp_mpi(cfg: dict):
    comm = get_comm()
    rank, world = rank_world(comm)

    set_seed(cfg["seed"] + rank)

    torch.set_num_threads(cfg["threads"]["num_threads"])
    torch.set_num_interop_threads(cfg["threads"]["interop_threads"])

    train_ds, test_ds, num_classes, _ = get_datasets(cfg)

    per_rank_bs = _effective_batch(cfg, world)
    sampler = ShardSampler(len(train_ds), rank=rank, world=world, shuffle=True, seed=cfg["seed"])
    train_loader = DataLoader(
        train_ds,
        batch_size=per_rank_bs,
        sampler=sampler,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=False
    )
    # Evaluate only on rank 0 to keep it simple
    test_loader = None
    if rank == 0:
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"],
                    weight_decay=cfg["train"]["weight_decay"])

    # make sure all ranks start with same params
    broadcast_model_params(comm, model, root=0)

    rows = []
    for epoch in range(cfg["train"]["epochs"]):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n = 0

        with Timer() as t:
            for i, (x, y) in enumerate(train_loader):
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

                # Allreduce gradients then average
                sync_gradients_avg(comm, model)

                opt.step()

                bs = x.size(0)
                epoch_loss += loss.item() * bs
                epoch_acc += accuracy(logits.detach(), y) * bs
                n += bs

        # Aggregate scalar metrics to rank 0 for logging (optional)
        # Simple approach: rank 0 uses its local stats (OK skeleton)
        if rank == 0:
            test_acc = evaluate(model, test_loader)
            rows.append([epoch, t.elapsed, epoch_loss / n, epoch_acc / n, test_acc])

    if rank == 0:
        out_dir = os.path.join(cfg["output"]["dir"], "metrics")
        ensure_dir(out_dir)
        write_csv(os.path.join(out_dir, f"dp_world_{world}.csv"),
                  rows,
                  header=["epoch", "epoch_time_sec", "train_loss", "train_acc", "test_acc"])

        ckpt_dir = os.path.join(cfg["output"]["dir"], "checkpoints")
        ensure_dir(ckpt_dir)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"dp_world_{world}.pt"))
