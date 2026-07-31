

import argparse
import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from smaformer_vit_baseline import SMAFormerBaseline

# ----------------------------------------------------------------------
# Reproducibility (Seeding)
# ----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------
def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_dice):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_dice": best_dice,
    }, path)

def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cuda")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["best_dice"]

# ----------------------------------------------------------------------
# Dataset (Unchanged)
# ----------------------------------------------------------------------
class ISIC2018Dataset(Dataset):
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, image_paths, mask_paths, img_size=512, train=True):
        assert len(image_paths) == len(mask_paths)
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.image_paths)

    def _load(self, path):
        from PIL import Image
        return Image.open(path)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image

        img = self._load(self.image_paths[idx]).convert("RGB").resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        mask = self._load(self.mask_paths[idx]).convert("L").resize(
            (self.img_size, self.img_size), Image.NEAREST
        )

        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()

        if self.train:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[1])
            angle = (torch.rand(1).item() * 2 - 1) * 15.0
            img = _rotate(img, angle, interpolation="bilinear")
            mask = _rotate(mask, angle, interpolation="nearest")
            mask = (mask > 0.5).float()

        img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return img, mask

def _rotate(tensor, angle_degrees, interpolation="bilinear"):
    import torchvision.transforms.functional as TF
    mode = TF.InterpolationMode.BILINEAR if interpolation == "bilinear" else TF.InterpolationMode.NEAREST
    return TF.rotate(tensor, angle_degrees, interpolation=mode)

# ----------------------------------------------------------------------
# Loss & Metrics (Unchanged)
# ----------------------------------------------------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice.mean()
        return bce_loss + dice_loss

@torch.no_grad()
def binary_seg_metrics(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds_flat = preds.reshape(preds.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)
    tp = (preds_flat * targets_flat).sum(dim=1)
    fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
    fn = ((1 - preds_flat) * targets_flat).sum(dim=1)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }

# ----------------------------------------------------------------------
# Train / validate loops (Unchanged)
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, criterion, device, accumulation_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for step, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(imgs)
            loss = criterion(logits, masks) / accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * accumulation_steps
    return running_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    agg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            logits = model(imgs)
            loss = criterion(logits, masks)
        running_loss += loss.item()
        m = binary_seg_metrics(logits, masks, threshold=threshold)
        for k in agg:
            agg[k] += m[k]
    n = len(loader)
    return running_loss / n, {k: v / n for k, v in agg.items()}

# ----------------------------------------------------------------------
# Main Loop (Upgraded)
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images", nargs="+", required=True)
    parser.add_argument("--train_masks", nargs="+", required=True)
    parser.add_argument("--val_images", nargs="+", required=True)
    parser.add_argument("--val_masks", nargs="+", required=True)

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--base_lr", type=float, default=1e-2)
    parser.add_argument("--vit_lr", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=6e-6)
    parser.add_argument("--momentum", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--vit_name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--pretrained_vit", action="store_true", default=True)
    parser.add_argument("--freeze_vit_blocks", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2) # LOWERED TO 2 FOR STABILITY
    
    # Checkpoints now point to Google Drive natively
    parser.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/checkpoints")
    args = parser.parse_args()

    set_seed(42) # Lock determinism
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ISIC2018Dataset(args.train_images, args.train_masks, img_size=args.img_size, train=True)
    val_ds = ISIC2018Dataset(args.val_images, args.val_masks, img_size=args.img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = SMAFormerBaseline(
        in_channels=3,
        n_classes=1,
        img_size=args.img_size,
        vit_name=args.vit_name,
        pretrained_vit=args.pretrained_vit,
        freeze_vit_blocks=args.freeze_vit_blocks,
    ).to(device)

    criterion = BCEDiceLoss()
    optimizer = torch.optim.SGD(
        model.param_groups(base_lr=args.base_lr, vit_lr=args.vit_lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Auto-Resume Logic
    start_epoch = 1
    best_dice = 0.0
    last_ckpt_path = os.path.join(args.out_dir, "last.pt")
    
    if os.path.exists(last_ckpt_path):
        print(f"🔄 Found interrupted run! Resuming from: {last_ckpt_path}")
        start_epoch, best_dice = load_checkpoint(last_ckpt_path, model, optimizer, scheduler, scaler)
        start_epoch += 1 # Start at the next epoch

    # Setup CSV Logger
    csv_path = os.path.join(args.out_dir, "training_log.csv")
    write_header = not os.path.exists(csv_path)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, args.accumulation_steps)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        curr_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f} | "
            f"lr={curr_lr:.6f}"
        )

        # 1. Log to CSV (Survives disconnects)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "lr"])
                write_header = False
            writer.writerow([epoch, train_loss, val_loss, val_metrics['dice'], val_metrics['iou'], val_metrics['precision'], val_metrics['recall'], curr_lr])

        # 2. Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pt"))
            print(f"🔥 New best Dice! Saved to best_model.pt")

        # 3. Save resume-state every epoch (Survives disconnects)
        save_checkpoint(last_ckpt_path, model, optimizer, scheduler, scaler, epoch, best_dice)

    print(f"Training complete. Best val Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()