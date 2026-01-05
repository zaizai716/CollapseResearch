import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
import numpy as np


class Wrapper(pl.LightningModule):

    def __init__(self, model, learning_rate=1e-5, epochs=5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tosave = False
        self.saved = []
        # Use manual optimization for full control over gradient updates
        self.automatic_optimization = False
        self.nan_count = 0
        self.total_steps = 0

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Ensure correct dimensions (batch_size, sequence_length)
        if len(input_ids.shape) != 2:
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)

        # Forward pass
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.total_steps += 1

        # Check for NaN/Inf loss - skip this batch entirely
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_count += 1
            if self.nan_count % 10 == 1:  # Only print every 10th to avoid spam
                print(f"WARNING: NaN/Inf loss at batch {batch_idx} (total: {self.nan_count}/{self.total_steps})")
            return None

        # Zero gradients
        opt.zero_grad()

        # Backward pass
        self.manual_backward(loss)

        # Check gradients for NaN/Inf BEFORE applying
        has_bad_grad = False
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_bad_grad = True
                    break

        if has_bad_grad:
            # Skip this update entirely - don't call optimizer.step()
            self.nan_count += 1
            if self.nan_count % 10 == 1:
                print(f"WARNING: Bad gradients at batch {batch_idx}, skipping update (total: {self.nan_count}/{self.total_steps})")
            opt.zero_grad()  # Clear the bad gradients
            return None

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Clip gradient values
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1.0, 1.0)

        # Only step if everything is good
        opt.step()

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Step the scheduler at the end of each epoch
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

        # Validate model weights at end of epoch
        nan_weights = 0
        inf_weights = 0
        total_weights = 0
        for name, param in self.named_parameters():
            total_weights += param.numel()
            nan_weights += torch.isnan(param).sum().item()
            inf_weights += torch.isinf(param).sum().item()

        if nan_weights > 0 or inf_weights > 0:
            print(f"CRITICAL: Model weights corrupted! NaN: {nan_weights}, Inf: {inf_weights}")
        else:
            print(f"Epoch end: Model weights healthy. Skipped {self.nan_count}/{self.total_steps} batches due to NaN/Inf")

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            labels = labels.unsqueeze(0)

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        # Handle NaN in validation
        if torch.isnan(loss) or torch.isinf(loss):
            return None

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            labels = labels.unsqueeze(0)

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        # Handle NaN in test
        if torch.isnan(loss) or torch.isinf(loss):
            return None

        perplexity = torch.exp(loss)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_perplexity", perplexity, on_step=False, on_epoch=True)

        if self.tosave:
            self.saved.append(perplexity.cpu().item())

        return {"loss": loss, "perplexity": perplexity}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01, eps=1e-6)

        # Warmup for first epoch, then linear decay
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=1
        )

        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.epochs - 1
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[1]
        )

        return [optimizer], [scheduler]
