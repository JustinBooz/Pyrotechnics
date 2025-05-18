"""
Train species-dist CNN with mixed-precision, AdamW, OneCycleLR, and freeze-unfreeze schedule.
This script orchestratess data loading, model setup, and training/validation loops.
"""
from __future__ import annotations

import os, logging, warnings
# suppress TF/Keras warnings if present
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")

import json, time
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from Data_Preparation_CNN import prepare_data, BLOCK_DEG, TRAIN_FRAC, VAL_FRAC, TEST_FRAC
from Model_Definition_CNN import ConvNeXt9Ch

# ────────── USER PATHS & HYPERPARAMS ──────────
CSV_PATH   = "path/to/iNaturalist_Complete_Project.csv"
NAIP_PATH  = "path/to/naip_2022.tif"
ELEV_PATH  = "path/to/elevation.tif"
SLOPE_PATH = "path/to/slope.tif"
ASPECT_PATH= "path/to/aspect.tif"
DISTW_PATH = "path/to/WD.tif"

PATCH_SIZE     = 128           # pixels per patch
BATCH_SIZE     = 32
NUM_EPOCHS     = 60
BASE_LR        = 1e-4          # backbone base LR
HEAD_LR        = 1e-3          # head (Linear) LR
WEIGHT_DECAY   = 0.05
FREEZE_EPOCHS  = 5             # freeze backbone for first N epochs
RUN_DIR        = "path/to/runs/eaton_aoi_convnext9"


def main():
    """
    main pipeline: set device, data loaders, model, optimizer, scheduler, and run trianing loops.
    """
    # device & logging setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=RUN_DIR)

    """
    data preparation step: load and wrap datasets into DataLoaders.
    """
    train_ds, val_ds, _, species_enc, norm_stats, _ = prepare_data(
        csv_path     = CSV_PATH,
        naip_path    = NAIP_PATH,
        cont_rasters = {
            "elev"  : ELEV_PATH,
            "slope" : SLOPE_PATH,
            "aspect": ASPECT_PATH,
            "distw" : DISTW_PATH,
        },
        patch_size   = PATCH_SIZE,
        block_deg    = BLOCK_DEG,
        train_frac   = TRAIN_FRAC,
        val_frac     = VAL_FRAC,
        test_frac    = TEST_FRAC
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    """
    model and optimizer setup: freeze backbone, set sepearate lrs for backbone and head.
    """
    model = ConvNeXt9Ch(num_classes=len(species_enc.classes_)).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": 0.0,      "weight_decay": WEIGHT_DECAY},
        {"params": model.head.parameters(),     "lr": HEAD_LR, "weight_decay": 0},
    ])

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[BASE_LR, HEAD_LR],
        total_steps=NUM_EPOCHS * len(train_loader),
        pct_start=FREEZE_EPOCHS / NUM_EPOCHS,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = GradScaler()

    """
    compute class weights for imbalnce dataset and define loss criterion.
    """
    counts = torch.as_tensor(train_ds.y, dtype=torch.long)
    class_counts = torch.bincount(counts)
    weights = (class_counts.float().mean() / class_counts.float()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    """
    training and validation loops: iterate epochs, update scheduler, save best model.
    """
    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        if epoch == FREEZE_EPOCHS + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer.param_groups[0]["lr"] = BASE_LR
            print(f"→ Unfroze backbone; backbone LR now {BASE_LR}")

        # train loop
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in tqdm(train_loader, desc="Train", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                logits = model(xb)
                loss   = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total   += yb.size(0)
        epoch_loss = train_loss / train_total
        epoch_acc  = train_correct / train_total
        print(f"Train ▶ loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

        # validation loop
        model.eval()
        val_loss = 0.0; val_correct = 0; val_total = 0
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc="Val  ", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss   = criterion(logits, yb)
                val_loss    += loss.item() * yb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total   += yb.size(0)
        val_loss /= val_total
        val_acc   = val_correct / val_total

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  epoch_acc,  epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)
        writer.add_scalar("LR/head", optimizer.param_groups[1]["lr"], epoch)

        elapsed = time.time() - epoch_start
        print(f"  train loss {epoch_loss:.4f} acc {epoch_acc:.2%} |"
              f" val loss {val_loss:.4f} acc {val_acc:.2%} ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(RUN_DIR) / "best_model.pt")

    """
    save artifacts: species encoder and norm stats to disk.
    """
    with open(Path(RUN_DIR) / "species_encoder.json", "w") as f:
        json.dump(species_enc.classes_.tolist(), f, indent=2)
    with open(Path(RUN_DIR) / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"✓ Finished: best val acc {best_val_acc:.2%}")
    print(f"  Artifacts in {RUN_DIR}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
