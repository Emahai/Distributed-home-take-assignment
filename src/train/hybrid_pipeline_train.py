import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.common.utils import set_seed, ensure_dir
from src.common.timers import Timer
from src.common.logging import write_csv
from src.common.mpitools import (
    get_comm, rank_world, split_by_color,
    send_tensor, recv_tensor, allreduce_tensor_sum_
)
from src.data.dataset import get_datasets
from src.data.sampler import ShardSampler
from src.models.cnn_stages import Stage0, Stage1, stage_output_shape
from src.common.correctness import accuracy

TAG_FWD = 100
TAG_BWD = 200

def _make_groups(comm, num_stages: int):
    rank = comm.Get_rank()
    world = comm.Get_size()
    assert world % num_stages == 0, "world_size must be divisible by num_stages"
    replica_id = rank // num_stages
    stage_id = rank % num_stages

    # communicator across ranks that share same stage_id (for gradient sync across replicas)
    stage_comm = split_by_color(comm, color=stage_id, key=replica_id)

    return replica_id, stage_id, stage_comm

def _effective_batch(cfg, world, num_stages):
    """
    Strategy: pipeline replicas = world/num_stages.
    Apply batch policy based on number of replicas (not total ranks).
    """
    replicas = world // num_stages
    bs = cfg["train"]["batch_size"]
    policy = cfg.get("strategy", {}).get("batch_policy", "global_fixed")
    if policy == "global_fixed":
        return max(1, bs // replicas)
    if policy == "per_rank_fixed":
        return bs
    raise ValueError(f"Unknown batch_policy: {policy}")

def _allreduce_stage_grads_avg(stage_comm, model_stage):
    world = stage_comm.Get_size()
    for p in model_stage.parameters():
        if p.grad is None:
            continue
        g = p.grad.data
        allreduce_tensor_sum_(stage_comm, g)
        g /= world

def run_hybrid_pipeline(cfg: dict):
    comm = get_comm()
    rank, world = rank_world(comm)

    num_stages = cfg["strategy"]["num_stages"]
    microbatches = cfg["strategy"]["microbatches"]
    assert num_stages == 2, "This skeleton implements 2-stage pipeline. Extend if needed."

    replica_id, stage_id, stage_comm = _make_groups(comm, num_stages=num_stages)

    set_seed(cfg["seed"] + rank)

    torch.set_num_threads(cfg["threads"]["num_threads"])
    torch.set_num_interop_threads(cfg["threads"]["interop_threads"])

    train_ds, _, num_classes, in_shape = get_datasets(cfg)

    per_replica_bs = _effective_batch(cfg, world, num_stages)
    assert per_replica_bs % microbatches == 0, "batch_size per replica must be divisible by microbatches"
    mb = per_replica_bs // microbatches

    # Data sharding happens across replicas (not all ranks)
    replicas = world // num_stages
    sampler = ShardSampler(len(train_ds), rank=replica_id, world=replicas, shuffle=True, seed=cfg["seed"])
    train_loader = DataLoader(
        train_ds,
        batch_size=per_replica_bs,
        sampler=sampler,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=False
    )

    # Build stage model
    if stage_id == 0:
        model = Stage0()
        out_c, out_h, out_w = stage_output_shape(in_shape)
    else:
        model = Stage1(num_classes=num_classes)
        out_c, out_h, out_w = stage_output_shape(in_shape)  # expected input channels
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"],
                    weight_decay=cfg["train"]["weight_decay"])

    # Determine stage neighbors inside replica
    # stage0 rank for this replica is replica_id*num_stages + 0
    # stage1 rank for this replica is replica_id*num_stages + 1
    stage0_rank = replica_id * num_stages + 0
    stage1_rank = replica_id * num_stages + 1

    rows = []
    for epoch in range(cfg["train"]["epochs"]):
        sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        n = 0

        with Timer() as t:
            for batch_idx, (x, y) in enumerate(train_loader):
                # Split into microbatches
                xs = torch.chunk(x, microbatches, dim=0)
                ys = torch.chunk(y, microbatches, dim=0)

                opt.zero_grad(set_to_none=True)

                for m_i in range(microbatches):
                    x_mb = xs[m_i].contiguous()
                    y_mb = ys[m_i].contiguous()

                    if stage_id == 0:
                        # Forward stage0
                        a0 = model(x_mb)          # (mb, 32, 16, 16)
                        a0.requires_grad_(True)

                        # Send activation to stage1
                        send_tensor(comm, a0.detach(), dest=stage1_rank, tag=TAG_FWD + m_i)

                        # Receive gradient w.r.t activation from stage1
                        da0 = recv_tensor(comm, shape=a0.shape, dtype=a0.detach().numpy().dtype,
                                          src=stage1_rank, tag=TAG_BWD + m_i)

                        # Backward through stage0 using received gradient
                        a0.backward(da0)

                    else:
                        # Stage1 receives activation
                        # shape known: (mb, 32, 16, 16)
                        a0 = recv_tensor(comm, shape=(mb, out_c, out_h, out_w),
                                         dtype=x_mb.detach().numpy().dtype,
                                         src=stage0_rank, tag=TAG_FWD + m_i)
                        a0.requires_grad_(True)

                        # Forward stage1 + loss
                        logits = model(a0)
                        loss = criterion(logits, y_mb)
                        loss.backward()

                        # Send gradient of activation back
                        send_tensor(comm, a0.grad.detach(), dest=stage0_rank, tag=TAG_BWD + m_i)

                        # Track correctness locally (only stage1 has logits)
                        bs = y_mb.size(0)
                        epoch_loss += loss.item() * bs
                        epoch_acc += accuracy(logits.detach(), y_mb) * bs
                        n += bs

                # Data-parallel sync per stage across replicas
                _allreduce_stage_grads_avg(stage_comm, model)

                # Update stage params
                opt.step()

        # Log only from stage1 ranks of replica 0 (cleanest single log)
        if (replica_id == 0) and (stage_id == 1):
            rows.append([epoch, t.elapsed, epoch_loss / max(n,1), epoch_acc / max(n,1)])

    if (replica_id == 0) and (stage_id == 1):
        out_dir = os.path.join(cfg["output"]["dir"], "metrics")
        ensure_dir(out_dir)
        write_csv(os.path.join(out_dir, f"hybrid_world_{world}_stages_{num_stages}.csv"),
                  rows,
                  header=["epoch", "epoch_time_sec", "train_loss", "train_acc"])

    # Save per-stage checkpoints for replica 0
    if replica_id == 0:
        ckpt_dir = os.path.join(cfg["output"]["dir"], "checkpoints")
        ensure_dir(ckpt_dir)
        torch.save(model.state_dict(),
                   os.path.join(ckpt_dir, f"hybrid_stage{stage_id}_world_{world}.pt"))
