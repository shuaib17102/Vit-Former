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

import albumentations as A
from albumentations.pytorch import ToTensorV2
from smaformer_core import replace_batchnorm_with_groupnorm

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
# Dataset (Upgraded with Albumentations)
# ----------------------------------------------------------------------
def get_isic_train_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),

        # --- mandatory tier ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),          # dermoscopy has no canonical "up"
        A.Affine(rotate=(-15, 15), scale=(0.9, 1.1), p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),

        # --- recommended upgrade tier ---
        A.ElasticTransform(alpha=1, sigma=20, p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),   # helps low-contrast / hair-obscured boundaries
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.02, 0.08),
                        hole_width_range=(0.02, 0.08), p=0.2),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_isic_val_transforms(img_size: int = 224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class ISIC2018Dataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import numpy as np
        from PIL import Image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        img_np = np.array(img)
        mask_np = np.array(mask, dtype=np.float32) / 255.0  

        if self.transform:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask'].unsqueeze(0) 
        else:
            import torchvision.transforms.functional as TF
            img_tensor = TF.to_tensor(img)
            mask_tensor = TF.to_tensor(mask)

        return img_tensor, mask_tensor

# ----------------------------------------------------------------------
# Loss & Metrics (Upgraded)
# ----------------------------------------------------------------------
class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 4.0, dice_smooth: float = 1.0, bce_dice_ratio: float = 0.5):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.dice_smooth = dice_smooth
        self.bce_dice_ratio = bce_dice_ratio

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1).float()
        targets_flat = targets.reshape(targets.size(0), -1).float()
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2 * intersection + self.dice_smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.dice_smooth
        )
        dice_loss = 1 - dice.mean()
        return self.bce_dice_ratio * bce + (1 - self.bce_dice_ratio) * dice_loss

def compute_pos_weight(mask_paths) -> float:
    import numpy as np
    from PIL import Image
    total_pos, total_pixels = 0, 0
    for p in mask_paths:
        m = np.array(Image.open(p).convert("L")) > 127
        total_pos += m.sum()
        total_pixels += m.size
    total_neg = total_pixels - total_pos
    return float(total_neg / max(total_pos, 1))

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1).float()
        targets_flat = targets.reshape(targets.size(0), -1).float()

        tp = (probs_flat * targets_flat).sum(dim=1)
        fn = ((1 - probs_flat) * targets_flat).sum(dim=1)
        fp = (probs_flat * (1 - targets_flat)).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss = (1 - tversky).pow(self.gamma)
        return loss.mean()

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
        
        # Explicit fp16 for T4 Tensor Cores
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
            logits = model(imgs)
            loss = criterion(logits, masks) / accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            # Unscale before gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        running_loss += loss.item() * accumulation_steps

    # FLUSH: Process any remaining gradients at the end of the epoch
    if len(loader) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    agg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
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
    parser.add_argument("--start_epoch", type=int, default=1)
    
    # Checkpoints now point to Google Drive natively
    parser.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/checkpoints")
    args = parser.parse_args()

    set_seed(42) # Lock determinism
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ISIC2018Dataset(args.train_images, args.train_masks, transform=get_isic_train_transforms(args.img_size))
    val_ds = ISIC2018Dataset(args.val_images, args.val_masks, transform=get_isic_val_transforms(args.img_size))
    
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
    )
    
    # Apply GroupNorm fix BEFORE sending to device
    model = replace_batchnorm_with_groupnorm(model, target_groups=8)
    model = model.to(device)

    # 3. Upgrade Criterion (Using FocalTversky as Claude recommended)
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75).to(device)

    optimizer = torch.optim.SGD(
        model.param_groups(base_lr=args.base_lr, vit_lr=args.vit_lr),
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Auto-Resume Logic
    start_epoch = args.start_epoch
    best_dice = 0.0
    last_ckpt_path = os.path.join(args.out_dir, "last.pt")
    
    if os.path.exists(last_ckpt_path):
        print(f"🔄 Found interrupted run! Resuming from: {last_ckpt_path}")
        start_epoch, best_dice = load_checkpoint(last_ckpt_path, model, optimizer, scheduler, scaler)
        start_epoch += 1

    # Setup CSV Logger
    csv_path = os.path.join(args.out_dir, "training_log.csv")
    write_header = not os.path.exists(csv_path)

    # Change the loop to calculate total remaining epochs correctly
    for epoch in range(start_epoch, start_epoch + args.epochs):
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
