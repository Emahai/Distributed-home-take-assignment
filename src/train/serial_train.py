import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.common.utils import set_seed, ensure_dir
from src.common.timers import Timer
from src.common.logging import write_csv
from src.data.dataset import get_datasets
from src.models.cnn_full import SimpleCNN
from src.train.evaluate import evaluate
from src.common.correctness import accuracy

def run_serial(cfg: dict):
    set_seed(cfg["seed"])

    torch.set_num_threads(cfg["threads"]["num_threads"])
    torch.set_num_interop_threads(cfg["threads"]["interop_threads"])

    train_ds, test_ds, num_classes, _ = get_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataset"]["num_workers"],
        pin_memory=False,
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=cfg["train"]["lr"], momentum=cfg["train"]["momentum"],
                    weight_decay=cfg["train"]["weight_decay"])

    rows = []
    for epoch in range(cfg["train"]["epochs"]):
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
                opt.step()

                bs = x.size(0)
                epoch_loss += loss.item() * bs
                epoch_acc += accuracy(logits.detach(), y) * bs
                n += bs

        test_acc = evaluate(model, test_loader)
        rows.append([epoch, t.elapsed, epoch_loss / n, epoch_acc / n, test_acc])

    out_dir = os.path.join(cfg["output"]["dir"], "metrics")
    ensure_dir(out_dir)
    write_csv(os.path.join(out_dir, "serial.csv"),
              rows,
              header=["epoch", "epoch_time_sec", "train_loss", "train_acc", "test_acc"])

    ckpt_dir = os.path.join(cfg["output"]["dir"], "checkpoints")
    ensure_dir(ckpt_dir)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "serial.pt"))
