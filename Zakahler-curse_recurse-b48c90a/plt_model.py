import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import numpy as np


class Wrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for language models to demonstrate model collapse.
    This implements the recursive training setup from "The Curse of Recursion" paper.
    """
    
    def __init__(self, model, learning_rate=1e-5, epochs=5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tosave = False
        self.saved = []
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the language model."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        """Training step - compute loss on training data."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        
        # Log training loss for monitoring
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - compute loss on validation data."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss
        
        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """
        Test step - compute loss and perplexity on test data.
        Perplexity is key metric for language model quality.
        Higher perplexity = worse model (more confused by data).
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
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
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        # Linear decay of learning rate over training
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=self.epochs
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }