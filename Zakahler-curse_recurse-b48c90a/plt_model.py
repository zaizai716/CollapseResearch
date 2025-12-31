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
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Ensure proper shapes and handle potential issues
        # Expected: [batch_size, sequence_length]
        
        # Check and print shapes for debugging
        if input_ids.shape[0] * input_ids.shape[1] == 8192 and labels is not None:
            if labels.shape[0] == 128:
                # This is our specific error case
                print(f"ERROR: Shape mismatch detected!")
                print(f"  input_ids: {input_ids.shape}")
                print(f"  labels: {labels.shape}")
                print(f"  Attempting to fix...")
                
                # The issue might be that input_ids is flattened but labels is not
                # Try to match dimensions
                if input_ids.dim() == 2 and labels.dim() == 2:
                    # Both are 2D, but different shapes
                    # Ensure they have the same shape
                    if input_ids.shape != labels.shape:
                        # Use input_ids shape as the correct one
                        labels = labels.reshape(input_ids.shape)
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Debug tensor shapes
        # print(f"DEBUG: input_ids shape: {input_ids.shape}, labels shape: {labels.shape}")
        
        # Ensure correct dimensions (batch_size, sequence_length)
        # The tensors should be 2D: [batch_size, sequence_length]
        if len(input_ids.shape) != 2:
            print(f"WARNING: Unexpected input_ids shape: {input_ids.shape}")
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                labels = labels.unsqueeze(0)
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        # Check for NaN/Inf and skip if detected
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected at batch {batch_idx}, skipping...")
            return None  # Skip this batch

        # Log training loss for monitoring
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Ensure correct dimensions (batch_size, sequence_length)
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            labels = labels.unsqueeze(0)
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Ensure correct dimensions (batch_size, sequence_length)
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            labels = labels.unsqueeze(0)
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        
        # Calculate perplexity - exponential of loss
        # This measures how "surprised" the model is by the test data
        perplexity = torch.exp(loss)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_perplexity", perplexity, on_step=False, on_epoch=True)
        
        if self.tosave:
            # Save individual perplexity scores for analysis
            self.saved.append(perplexity.cpu().item())
        
        return {"loss": loss, "perplexity": perplexity}
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        # Warmup for first epoch, then linear decay
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of lr
            end_factor=1.0,    # Warmup to 100%
            total_iters=1      # Warmup over 1 epoch
        )

        # Linear decay after warmup
        decay_scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.epochs - 1
        )

        # Combine warmup + decay
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[1]  # Switch after epoch 1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def on_before_optimizer_step(self, optimizer):
        # Check for NaN/Inf in gradients and zero them out if found
        has_nan_grad = False
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    param.grad.zero_()  # Zero out bad gradients instead of applying them

        if has_nan_grad:
            print("WARNING: NaN/Inf gradients detected and zeroed out")

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Also clip gradient values (not just norm) to prevent extreme values
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1.0, 1.0)