# trainer_core.py -- Trainer with loss-component logging.

import os, torch, yaml, wandb
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

class Trainer:
    def __init__(self, cfg_path: str, build_components_fn):
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        (self.model, self.criterion, self.optimizer, self.dl_train, self.dl_val, self.misc) = build_components_fn(self.cfg, self.device)

        # Configure linear warmup and cosine learning-rate schedules.
        warm, total = 5, self.cfg["epochs"]
        self.sched_warm = LambdaLR(self.optimizer, lr_lambda=lambda e: min(1., (e+1)/warm))
        self.sched_cos  = CosineAnnealingLR(self.optimizer, T_max=total-warm, eta_min=1e-6)
        self.scaler = GradScaler()
        self.ema    = ExponentialMovingAverage(self.model.parameters(), 0.999)

        self.start_epoch = 0
        self.best_loss   = float("inf")
        self.best_f1     = 0.0
        self.wait        = 0

        self.base_save_path = self.cfg["save_path"]
        self.save_path = self.base_save_path

        if self.cfg.get("resume"):
            resume_path = self.base_save_path.replace(".pth", "_best_loss.pth")
            if os.path.exists(resume_path):
                self._resume_ckpt(resume_path)
                self.save_path = resume_path

        if self.cfg.get("log_with_wandb"):
            wandb.init(project=self.cfg["wandb_project"], config=self.cfg)

    def _resume_ckpt(self, path=None):
        ckpt = torch.load(path or self.save_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.sched_warm.load_state_dict(ckpt["sched_warm"])
        self.sched_cos .load_state_dict(ckpt["sched_cos"])
        self.scaler    .load_state_dict(ckpt["scaler"])
        self.ema       .load_state_dict(ckpt["ema"])
        self.start_epoch = ckpt["epoch"]

    def _save_ckpt(self, epoch: int, path=None):
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sched_warm": self.sched_warm.state_dict(),
            "sched_cos": self.sched_cos.state_dict(),
            "scaler": self.scaler.state_dict(),
            "ema": self.ema.state_dict(),
            **self.misc
        }, path or self.save_path)
        tqdm.write(f"✅ Model saved to {path or self.save_path}")

    def fit(self, validate_fn):
        total = self.cfg["epochs"]
        patience = self.cfg["patience"]

        for epoch in range(self.start_epoch, total):
            self.model.train()
            train_loss, step = 0.0, 0
            loss_accum = {"ciou_loss": 0.0, "obj_loss": 0.0, "cls_loss": 0.0}

            for batch in self.dl_train:
                if batch is None: continue
                loss, loss_details = self._train_step(batch)
                train_loss += loss.item()
                for k in loss_accum:
                    if k in loss_details:
                        loss_accum[k] += loss_details[k]
                step += 1

            avg = train_loss / max(step, 1)
            avg_losses = {k: v / max(step, 1) for k, v in loss_accum.items()}

            print(f"[Epoch {epoch+1}] Train Loss {avg:.4f} (CIoU: {avg_losses['ciou_loss']:.4f}, Obj: {avg_losses['obj_loss']:.4f}, Cls: {avg_losses['cls_loss']:.4f})")
            (self.sched_cos if epoch+1 > 3 else self.sched_warm).step()

            # ✅ Save best_loss checkpoint
            if avg < self.best_loss:
                self.best_loss = avg; self.wait = 0
                path = self.base_save_path.replace(".pth", "_best_loss.pth")
                self._save_ckpt(epoch+1, path)
            else:
                self.wait += 1
                if self.wait >= patience:
                    print("⚠️  Early‑Stopping"); break

            # ✅ Validate every N epochs
            if (epoch+1) % self.cfg.get("val_interval", 1) == 0:
                prec, rec, f1, ap = validate_fn(self.model, self.ema, self.dl_val, self.device)
                # prec, rec, f1, ap = validate_fn(self.model, None, self.dl_val, self.device)

                print(f"[Epoch {epoch+1}] P: {prec:.4f} R: {rec:.4f} F1: {f1:.4f} AP: {ap:.4f}")

                # ✅ Save best_f1 checkpoint
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    path = self.base_save_path.replace(".pth", "_best_f1.pth")
                    self._save_ckpt(epoch+1, path)

                if self.cfg.get("log_with_wandb"):
                    wandb.log({"train_loss": avg, **avg_losses, "precision": prec, "recall": rec, "f1": f1, "ap": ap, "epoch": epoch+1})
            elif self.cfg.get("log_with_wandb"):
                wandb.log({"train_loss": avg, **avg_losses, "epoch": epoch+1})

        # ✅ Final validation + PR Curve
        prec, rec, f1, ap = validate_fn(self.model, self.ema, self.dl_val, self.device, save_pr_curve=True)
        print(f"[Final] P: {prec:.4f} R: {rec:.4f} F1: {f1:.4f} AP: {ap:.4f}")
        self._save_ckpt(total, self.base_save_path.replace(".pth", "_final.pth"))

    def _train_step(self, batch):
        self.optimizer.zero_grad()
        with autocast(enabled=self.cfg["use_amp"]):
            out = self.model.forward_batch(batch, self.device)

            if len(out) == 2:  # Early / feature fusion
                preds, targets = out
                loss, loss_details = self.criterion(preds, targets)
            else:  # Result-level fusion
                p_rgb, p_lidar, targets = out
                loss, loss_details = self.criterion(p_rgb, p_lidar, targets)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update()

        return loss, loss_details
