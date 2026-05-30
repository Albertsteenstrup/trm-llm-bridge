"""
train_translator_bridge.py — Path 1: Trainable Q-Former bridge with constrained output

Trains a lightweight cross-attention module (81 learnable queries, one per grid cell)
that reads frozen LLM hidden states and classifies each cell as empty (.) or digit (1-9).

Architecture (inspired by BLIP-2 Q-Former):
  - Frozen LLM encodes the NL description → token-level hidden states
  - 81 learnable query vectors cross-attend to those hidden states
  - Self-attention between queries captures row/column/box dependencies
  - Per-query linear head → 10-class softmax (0=empty, 1-9=digit)
  - Constrained output layer guarantees valid 81-char grid string

References:
  - Li et al., "BLIP-2", ICML 2023, arXiv:2301.12597

Usage:
    python code/initial/integration/train_translator_bridge.py
    python code/initial/integration/train_translator_bridge.py --epochs 50 --given-weight 3.0
    python code/initial/integration/train_translator_bridge.py --patience 10 --batch-size 16

    # On HPC via Slurm:
    sbatch scripts/initial/translators/train_bridge.sbatch
"""

import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

@dataclass
class BridgeConfig:
    """All hyperparameters for the Q-Former bridge translator."""

    # --- LLM ---
    llm_model_name: str = "checkpoints/initial/Qwen3-1.7B"  # resolved to repo root if relative
    llm_hidden_size: int = 0             # 0 = auto-detect from model config after loading
    llm_max_seq_len: int = 512           # max tokens for NL input
    llm_layer_index: int = -1            # which hidden-state layer to tap (-1 = last)
    llm_dtype: str = "auto"              # auto -> bf16 on supported GPUs, fp16 otherwise

    # --- Bridge (Q-Former) ---
    num_queries: int = 81                # one per grid cell
    bridge_dim: int = 256                # internal dimension of bridge module
    bridge_heads: int = 4                # attention heads
    bridge_layers: int = 4              # number of cross-attn + self-attn layers
    bridge_dropout: float = 0.1
    num_classes: int = 10                # 0=empty, 1-9=digit
    transnar_recurrence_steps: int = 1   # extra tied passes for TransNAR-style interleaving

    # --- Training ---
    dataset_path: str = "data/initial/sudoku_synthetic/rule/sudoku_nl_diverse.json"
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 50
    warmup_fraction: float = 0.05       # fraction of total steps for LR warmup
    eval_split: float = 0.1             # fraction held out for validation
    seed: int = 42
    save_every_epoch: int = 5
    patience: int = 8                    # early stopping: epochs without val improvement
    given_class_weight: float = 3.0      # loss weight for digit classes 1-9 vs empty (0)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths ---
    checkpoint_dir: str = "checkpoints/initial/translator_bridge"
    log_dir: str = "logs/initial"


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

class SudokuNLDataset(Dataset):
    """
    Loads NL→grid pairs. Each sample returns tokenized NL text and an 81-long
    target vector where each position is 0 (empty) or 1-9 (digit).
    """

    def __init__(self, records: list, tokenizer, max_length: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        nl_text = extract_nl_text(item)

        # Parse puzzle string → per-cell integer targets
        puzzle_str = item["puzzle"]
        target = []
        for ch in puzzle_str:
            if ch == ".":
                target.append(0)
            elif ch in "123456789":
                target.append(int(ch))
            else:
                target.append(0)

        inputs = self.tokenizer(
            nl_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "target_grid": torch.tensor(target, dtype=torch.long),
            "puzzle_str": puzzle_str,
        }


def extract_nl_text(record: dict) -> str:
    """Best-effort extraction of text input used to condition the bridge model."""
    for key in ("nl_description", "description", "prompt", "text", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return ""


# ---------------------------------------------------------------------------
# 3. Model: Q-Former Bridge
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer: queries attend to LLM hidden states."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, queries, kv, kv_mask=None):
        # Cross-attention: queries attend to LLM hidden states
        # kv_mask: (batch, seq_len) bool, True = valid token
        key_padding_mask = ~kv_mask if kv_mask is not None else None
        x = queries
        attn_out, _ = self.cross_attn(x, kv, kv, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)

        # Self-attention: queries attend to each other (captures grid structure)
        self_out, _ = self.self_attn(x, x, x)
        x = self.norm2(x + self_out)

        # Feed-forward
        x = self.norm3(x + self.ffn(x))
        return x


class QFormerBridge(nn.Module):
    """
    81 learnable queries cross-attend to frozen LLM hidden states,
    then a per-query classification head predicts 10 classes per cell.

    Parameter count (default config):
      - Input projection: 2048 × 256 = ~524K
      - 81 queries: 81 × 256 = ~21K
      - Positional embeddings: 81 × 256 = ~21K
      - 4 layers × ~1.3M each = ~5.3M
      - Classification head: 256 × 10 = ~2.6K
      Total: ~6M parameters (well under 100M limit)
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config

        # Project LLM hidden states to bridge dimension
        self.input_proj = nn.Linear(config.llm_hidden_size, config.bridge_dim)

        # Learnable query vectors (one per grid cell)
        self.queries = nn.Parameter(torch.randn(config.num_queries, config.bridge_dim) * 0.02)

        # Positional embeddings encoding (row, col, box) structure
        self.pos_embed = nn.Parameter(torch.zeros(config.num_queries, config.bridge_dim))
        self._init_positional_embeddings()

        # Cross-attention + self-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(config.bridge_dim, config.bridge_heads, config.bridge_dropout)
            for _ in range(config.bridge_layers)
        ])

        # Classification head: predict 10 classes per cell
        self.classifier = nn.Linear(config.bridge_dim, config.num_classes)

    def _init_positional_embeddings(self):
        """Initialize positional embeddings with row/col/box structure."""
        dim = self.config.bridge_dim
        third = dim // 3

        with torch.no_grad():
            for cell in range(81):
                row = cell // 9
                col = cell % 9
                box = (row // 3) * 3 + (col // 3)

                # Sinusoidal encoding for row, col, box
                for i in range(third):
                    freq = 1.0 / (10000.0 ** (2.0 * i / third))
                    self.pos_embed.data[cell, i] = math.sin(row * freq)
                    self.pos_embed.data[cell, third + i] = math.sin(col * freq)
                    if 2 * third + i < dim:
                        self.pos_embed.data[cell, 2 * third + i] = math.sin(box * freq)

    def forward(self, llm_hidden_states, attention_mask=None):
        """
        Args:
            llm_hidden_states: (batch, seq_len, llm_hidden_size) from frozen LLM
            attention_mask: (batch, seq_len) bool mask for valid tokens

        Returns:
            logits: (batch, 81, 10) per-cell class logits
        """
        batch_size = llm_hidden_states.shape[0]

        # Project LLM hidden states to bridge dimension
        kv = self.input_proj(llm_hidden_states)  # (batch, seq_len, bridge_dim)

        # Expand queries + positional embeddings for the batch
        queries = (self.queries + self.pos_embed).unsqueeze(0).expand(batch_size, -1, -1)

        # Pass through cross-attention + self-attention layers
        for layer in self.layers:
            queries = layer(queries, kv, kv_mask=attention_mask)

        # Classify each cell
        logits = self.classifier(queries)  # (batch, 81, 10)
        return logits

    def predict_grid(self, llm_hidden_states, attention_mask=None) -> list[str]:
        """Run inference and return constrained grid strings."""
        logits = self.forward(llm_hidden_states, attention_mask)
        return constrained_decode(logits)


class TransNARTRMInterleaveLayer(nn.Module):
    """
    TransNAR-style gated interleave block.

    The TRM state stream is represented as 81 solver-facing cell states. This keeps
    the initial Sudoku path self-contained while matching the key TransNAR pattern:
    token states and structured states update each other through zero-init gated
    cross-attention. If real frozen TRM latents become available later, they can
    replace or initialize the `trm_states` stream without changing the outer API.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.trm_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.trm_cross_norm_q = nn.LayerNorm(dim)
        self.trm_cross_norm_kv = nn.LayerNorm(dim)
        self.trm_self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.trm_self_norm = nn.LayerNorm(dim)
        self.trm_ffn_norm = nn.LayerNorm(dim)
        self.trm_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.token_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.token_cross_norm_q = nn.LayerNorm(dim)
        self.token_cross_norm_kv = nn.LayerNorm(dim)
        self.token_ffn_norm = nn.LayerNorm(dim)
        self.token_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Zero-init gates follow the TransNAR/Flamingo-style training trick.
        self.trm_cross_gate = nn.Parameter(torch.zeros(()))
        self.trm_self_gate = nn.Parameter(torch.zeros(()))
        self.trm_ffn_gate = nn.Parameter(torch.zeros(()))
        self.token_cross_gate = nn.Parameter(torch.zeros(()))
        self.token_ffn_gate = nn.Parameter(torch.zeros(()))

    def forward(self, trm_states, token_states, token_mask=None):
        key_padding_mask = ~token_mask if token_mask is not None else None

        trm_q = self.trm_cross_norm_q(trm_states)
        token_kv = self.trm_cross_norm_kv(token_states)
        trm_cross_out, _ = self.trm_cross_attn(
            trm_q, token_kv, token_kv, key_padding_mask=key_padding_mask
        )
        trm_states = trm_states + torch.tanh(self.trm_cross_gate) * trm_cross_out

        trm_self_in = self.trm_self_norm(trm_states)
        trm_self_out, _ = self.trm_self_attn(trm_self_in, trm_self_in, trm_self_in)
        trm_states = trm_states + torch.tanh(self.trm_self_gate) * trm_self_out

        trm_ffn_in = self.trm_ffn_norm(trm_states)
        trm_states = trm_states + torch.tanh(self.trm_ffn_gate) * self.trm_ffn(trm_ffn_in)

        token_q = self.token_cross_norm_q(token_states)
        trm_kv = self.token_cross_norm_kv(trm_states)
        token_cross_out, _ = self.token_cross_attn(token_q, trm_kv, trm_kv)
        token_states = token_states + torch.tanh(self.token_cross_gate) * token_cross_out

        token_ffn_in = self.token_ffn_norm(token_states)
        token_states = token_states + torch.tanh(self.token_ffn_gate) * self.token_ffn(token_ffn_in)
        return trm_states, token_states


class TransNARTRMBridge(nn.Module):
    """
    TransNAR-inspired Qwen<->TRM bridge for the initial Sudoku path.

    Compared with QFormerBridge, this model keeps a solver-facing 81-cell state
    stream and updates both token states and cell states through gated
    cross-attention. It does not expose real frozen TRM internals yet; instead it
    provides the same architectural slot where real TRM latent states can later be
    inserted.
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.llm_hidden_size, config.bridge_dim)
        self.queries = nn.Parameter(torch.randn(config.num_queries, config.bridge_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(config.num_queries, config.bridge_dim))
        self._init_positional_embeddings()
        self.layers = nn.ModuleList([
            TransNARTRMInterleaveLayer(
                dim=config.bridge_dim,
                num_heads=config.bridge_heads,
                dropout=config.bridge_dropout,
            )
            for _ in range(config.bridge_layers)
        ])
        self.classifier = nn.Linear(config.bridge_dim, config.num_classes)

    def _init_positional_embeddings(self):
        dim = self.config.bridge_dim
        third = dim // 3

        with torch.no_grad():
            for cell in range(81):
                row = cell // 9
                col = cell % 9
                box = (row // 3) * 3 + (col // 3)

                for i in range(third):
                    freq = 1.0 / (10000.0 ** (2.0 * i / third))
                    self.pos_embed.data[cell, i] = math.sin(row * freq)
                    self.pos_embed.data[cell, third + i] = math.sin(col * freq)
                    if 2 * third + i < dim:
                        self.pos_embed.data[cell, 2 * third + i] = math.sin(box * freq)

    def forward(self, llm_hidden_states, attention_mask=None):
        batch_size = llm_hidden_states.shape[0]
        token_states = self.input_proj(llm_hidden_states)
        trm_states = (self.queries + self.pos_embed).unsqueeze(0).expand(batch_size, -1, -1)

        recurrence_steps = max(1, int(self.config.transnar_recurrence_steps))
        for _ in range(recurrence_steps):
            for layer in self.layers:
                trm_states, token_states = layer(
                    trm_states=trm_states,
                    token_states=token_states,
                    token_mask=attention_mask,
                )

        logits = self.classifier(trm_states)
        return logits

    def predict_grid(self, llm_hidden_states, attention_mask=None) -> list[str]:
        logits = self.forward(llm_hidden_states, attention_mask)
        return constrained_decode(logits)


# ---------------------------------------------------------------------------
# 4. Constrained Output Layer
# ---------------------------------------------------------------------------

def constrained_decode(logits: torch.Tensor) -> list[str]:
    """
    Convert (batch, 81, 10) logits to valid 81-char grid strings.
    Guarantees output matches regex [1-9.]{81}.

    Each position: argmax over 10 classes → 0 maps to '.', 1-9 map to '1'-'9'.
    """
    preds = logits.argmax(dim=-1)  # (batch, 81)
    grids = []
    for row in preds.tolist():
        chars = []
        for val in row:
            if val == 0:
                chars.append(".")
            elif 1 <= val <= 9:
                chars.append(str(val))
            else:
                chars.append(".")  # safety fallback
        grids.append("".join(chars))
    return grids


def validate_grid_string(grid: str) -> bool:
    """Check that a grid string is structurally valid for TRM input."""
    if len(grid) != 81:
        return False
    return all(ch == "." or ch in "123456789" for ch in grid)


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_grids: list[str], target_grids: list[str]) -> dict:
    """
    Compute translator metrics:
      - cell_accuracy: fraction of correctly predicted cells across all samples
      - exact_match: fraction of samples where all 81 cells are correct
      - given_accuracy: accuracy only on non-empty (given) cells
      - empty_accuracy: accuracy only on empty cells
      - format_valid: fraction of outputs that are valid 81-char grid strings
    """
    total_cells = 0
    correct_cells = 0
    exact_matches = 0
    given_total = 0
    given_correct = 0
    empty_total = 0
    empty_correct = 0
    format_valid = 0

    for pred, target in zip(pred_grids, target_grids):
        if validate_grid_string(pred):
            format_valid += 1

        is_exact = True
        for p_ch, t_ch in zip(pred, target):
            total_cells += 1
            if p_ch == t_ch:
                correct_cells += 1
            else:
                is_exact = False

            if t_ch != ".":
                given_total += 1
                if p_ch == t_ch:
                    given_correct += 1
            else:
                empty_total += 1
                if p_ch == t_ch:
                    empty_correct += 1

        if is_exact:
            exact_matches += 1

    n = max(len(pred_grids), 1)
    return {
        "cell_accuracy": correct_cells / max(total_cells, 1),
        "exact_match": exact_matches / n,
        "given_accuracy": given_correct / max(given_total, 1),
        "empty_accuracy": empty_correct / max(empty_total, 1),
        "format_valid_rate": format_valid / n,
    }


# ---------------------------------------------------------------------------
# 6. Class-Weighted Loss
# ---------------------------------------------------------------------------

def build_class_weights(config: BridgeConfig) -> torch.Tensor:
    """
    Build per-class weights for CrossEntropyLoss to counteract empty-cell dominance.

    In a typical Sudoku puzzle, ~56 of 81 cells are empty (class 0 ≈ 69%)
    and ~25 are givens (classes 1-9 ≈ 31% combined, ~3.4% each).
    Without weighting the model learns to predict all-empty grids first
    because predicting '.' everywhere already achieves ~69% cell accuracy.

    Weight vector: class 0 gets weight 1.0, classes 1-9 get `given_class_weight`.
    """
    weights = torch.ones(config.num_classes, device=config.device)
    weights[1:] = config.given_class_weight
    return weights


# ---------------------------------------------------------------------------
# 7. Training Loop
# ---------------------------------------------------------------------------

def get_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 10% of peak (not zero)."""
    min_lr_ratio = 0.1  # floor at 10% of peak LR to avoid late-training stalling

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    bridge: QFormerBridge,
    llm,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    criterion: nn.CrossEntropyLoss,
    config: BridgeConfig,
    epoch: int,
):
    bridge.train()
    total_loss = 0.0
    total_correct = 0
    total_cells = 0
    given_correct = 0
    given_total = 0

    pbar = tqdm(dataloader, desc=f"Train epoch {epoch + 1}/{config.epochs}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        target_grid = batch["target_grid"].to(config.device)

        # Forward through frozen LLM
        with torch.no_grad():
            outputs = llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[config.llm_layer_index]

        # Forward through bridge
        optimizer.zero_grad()
        logits = bridge(hidden_states.float(), attention_mask.bool())  # (B, 81, 10)

        loss = criterion(logits.reshape(-1, config.num_classes), target_grid.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == target_grid).sum().item()
        total_cells += target_grid.numel()

        # Track given-cell accuracy during training (the metric that matters)
        given_mask = target_grid > 0
        if given_mask.any():
            given_correct += (preds[given_mask] == target_grid[given_mask]).sum().item()
            given_total += given_mask.sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cell_acc": f"{total_correct / total_cells:.4f}",
            "given_acc": f"{given_correct / max(given_total, 1):.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    avg_loss = total_loss / max(len(dataloader), 1)
    cell_acc = total_correct / max(total_cells, 1)
    gvn_acc = given_correct / max(given_total, 1)
    return avg_loss, cell_acc, gvn_acc


@torch.no_grad()
def evaluate(bridge: QFormerBridge, llm, dataloader: DataLoader, config: BridgeConfig):
    bridge.eval()
    total_loss = 0.0
    # Use unweighted loss for val so metrics are comparable across weight configs
    criterion = nn.CrossEntropyLoss()
    all_pred_grids = []
    all_target_grids = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        target_grid = batch["target_grid"].to(config.device)
        puzzle_strs = batch["puzzle_str"]

        outputs = llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[config.llm_layer_index]

        logits = bridge(hidden_states.float(), attention_mask.bool())
        loss = criterion(logits.reshape(-1, config.num_classes), target_grid.reshape(-1))
        total_loss += loss.item()

        pred_grids = constrained_decode(logits)
        all_pred_grids.extend(pred_grids)
        all_target_grids.extend(puzzle_strs)

    avg_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_metrics(all_pred_grids, all_target_grids)
    metrics["loss"] = avg_loss
    return metrics, all_pred_grids, all_target_grids


# ---------------------------------------------------------------------------
# 8. Error Analysis
# ---------------------------------------------------------------------------

def error_analysis(pred_grids: list[str], target_grids: list[str], max_examples: int = 20) -> dict:
    """
    Categorize dominant failure modes:
      - missing_given: target has digit, pred has '.'
      - hallucinated_given: target has '.', pred has digit
      - wrong_value: both non-empty but different digits
    """
    categories = {"missing_given": 0, "hallucinated_given": 0, "wrong_value": 0, "correct": 0}
    total_errors = 0
    error_examples = []

    for pred, target in zip(pred_grids, target_grids):
        sample_errors = []
        for pos, (p, t) in enumerate(zip(pred, target)):
            row = pos // 9 + 1
            col = pos % 9 + 1
            if p == t:
                categories["correct"] += 1
            elif t != "." and p == ".":
                categories["missing_given"] += 1
                sample_errors.append({"type": "missing_given", "row": row, "col": col, "expected": t})
                total_errors += 1
            elif t == "." and p != ".":
                categories["hallucinated_given"] += 1
                sample_errors.append({"type": "hallucinated_given", "row": row, "col": col, "predicted": p})
                total_errors += 1
            else:
                categories["wrong_value"] += 1
                sample_errors.append({"type": "wrong_value", "row": row, "col": col, "expected": t, "predicted": p})
                total_errors += 1

        if sample_errors and len(error_examples) < max_examples:
            error_examples.append({
                "target": target,
                "predicted": pred,
                "errors": sample_errors[:5],
            })

    return {
        "error_counts": categories,
        "total_errors": total_errors,
        "error_examples": error_examples,
    }


# ---------------------------------------------------------------------------
# 9. Baselines (Linear Probe & MLP)
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """
    Minimal baseline: single linear projection from pooled LLM hidden states.
    ~3.3M params for hidden_size=2048, grid_size=81, num_classes=10.

    Reference: "Mysterious Projections" (arXiv:2402.16832, 2024)
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        self.pool_proj = nn.Linear(config.llm_hidden_size, config.num_queries * config.num_classes)
        self.num_queries = config.num_queries
        self.num_classes = config.num_classes

    def forward(self, llm_hidden_states, attention_mask=None):
        # Mean pool over non-padding tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (llm_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = llm_hidden_states.mean(dim=1)

        logits = self.pool_proj(pooled)  # (batch, 81 * 10)
        return logits.view(-1, self.num_queries, self.num_classes)

    def predict_grid(self, llm_hidden_states, attention_mask=None) -> list[str]:
        logits = self.forward(llm_hidden_states, attention_mask)
        return constrained_decode(logits)


class MLPProjector(nn.Module):
    """
    LLaVA-style 2-layer MLP projector from pooled LLM states.
    ~10M params with default config.

    Reference: Liu et al., "LLaVA-1.5", arXiv:2310.03744
    """

    def __init__(self, config: BridgeConfig):
        super().__init__()
        intermediate = config.llm_hidden_size
        self.projector = nn.Sequential(
            nn.Linear(config.llm_hidden_size, intermediate),
            nn.GELU(),
            nn.Dropout(config.bridge_dropout),
            nn.Linear(intermediate, config.num_queries * config.num_classes),
        )
        self.num_queries = config.num_queries
        self.num_classes = config.num_classes

    def forward(self, llm_hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (llm_hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = llm_hidden_states.mean(dim=1)

        logits = self.projector(pooled)
        return logits.view(-1, self.num_queries, self.num_classes)

    def predict_grid(self, llm_hidden_states, attention_mask=None) -> list[str]:
        logits = self.forward(llm_hidden_states, attention_mask)
        return constrained_decode(logits)


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------

ARCH_REGISTRY = {
    "qformer": QFormerBridge,
    "transnar_trm": TransNARTRMBridge,
    "linear": LinearProbe,
    "mlp": MLPProjector,
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint_any(path: str, device: str):
    """Load checkpoint object (supports model-only and full training-state checkpoints)."""
    return torch.load(path, map_location=device)


def extract_model_state_dict(checkpoint_obj):
    """Extract model state_dict from either full checkpoint dict or plain state_dict."""
    if isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj:
        return checkpoint_obj["model_state_dict"]
    return checkpoint_obj


def extract_checkpoint_arch(checkpoint_obj, default: str = "qformer") -> str:
    """Infer bridge architecture from a saved checkpoint payload."""
    if isinstance(checkpoint_obj, dict):
        arch = checkpoint_obj.get("arch")
        if isinstance(arch, str) and arch in ARCH_REGISTRY:
            return arch
    return default


def extract_bridge_config_from_checkpoint(checkpoint_obj) -> Optional[BridgeConfig]:
    """Recover saved BridgeConfig when available."""
    if not isinstance(checkpoint_obj, dict):
        return None

    payload = checkpoint_obj.get("bridge_config")
    if not isinstance(payload, dict):
        return None

    cfg = BridgeConfig()
    for key, value in payload.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def build_model_checkpoint_payload(bridge: nn.Module, config: BridgeConfig, arch: str, **extra):
    payload = {
        "arch": arch,
        "bridge_config": asdict(config),
        "model_state_dict": bridge.state_dict(),
    }
    payload.update(extra)
    return payload


def infer_epoch_from_checkpoint_path(path: str) -> Optional[int]:
    """Infer epoch from checkpoint filename pattern like *_epoch65.pt."""
    match = re.search(r"_epoch(\d+)\.pt$", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def resolve_torch_dtype(dtype_name: str, device: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported --llm-dtype '{dtype_name}'")
    return mapping[dtype_name]


def main():
    parser = argparse.ArgumentParser(description="Train Q-Former bridge translator (Path 1)")
    parser.add_argument("--arch", choices=list(ARCH_REGISTRY.keys()), default="qformer",
                        help="Bridge architecture: qformer, transnar_trm, mlp, or linear")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--given-weight", type=float, default=None,
                        help="Override class weight for digit classes 1-9 (default: 3.0)")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Override frozen LLM path or Hugging Face model ID")
    parser.add_argument("--llm-dtype", type=str, default=None,
                        help="LLM load dtype: auto, bfloat16, float16, or float32")
    parser.add_argument("--llm-layer", type=int, default=None, help="Override LLM layer index to tap")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint directory")
    parser.add_argument("--eval-only", type=str, default=None, help="Path to checkpoint for eval-only mode")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from checkpoint path (full-state preferred)")
    parser.add_argument("--no-auto-resume", action="store_true",
                        help="Disable auto-resume from checkpoint_dir/last_<arch>.pt")
    parser.add_argument("--transnar-recurrence-steps", type=int, default=None,
                        help="Extra tied interleave passes for arch=transnar_trm")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="trm-llm-translator-bridge",
                        help="W&B project name (used when --wandb is set)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="Optional W&B run name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="Optional W&B entity/team")
    args = parser.parse_args()

    config = BridgeConfig()
    if args.dataset:
        config.dataset_path = args.dataset
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.patience is not None:
        config.patience = args.patience
    if args.given_weight is not None:
        config.given_class_weight = args.given_weight
    if args.llm_model:
        config.llm_model_name = args.llm_model
    if args.llm_dtype:
        config.llm_dtype = args.llm_dtype
    if args.llm_layer is not None:
        config.llm_layer_index = args.llm_layer
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.transnar_recurrence_steps is not None:
        config.transnar_recurrence_steps = args.transnar_recurrence_steps

    # Resolve relative paths from repo root
    repo_root = Path(__file__).resolve().parents[3]
    if not Path(config.dataset_path).is_absolute():
        config.dataset_path = str((repo_root / config.dataset_path).resolve())
    if not Path(config.checkpoint_dir).is_absolute():
        config.checkpoint_dir = str((repo_root / config.checkpoint_dir).resolve())
    if not Path(config.log_dir).is_absolute():
        config.log_dir = str((repo_root / config.log_dir).resolve())
    # Resolve LLM path: use local dir if it exists, otherwise treat as HF hub ID
    if not Path(config.llm_model_name).is_absolute():
        resolved_llm = (repo_root / config.llm_model_name).resolve()
        if resolved_llm.exists():
            config.llm_model_name = str(resolved_llm)

    torch.manual_seed(config.seed)

    # --- Load LLM (frozen) ---
    print(f"Loading frozen LLM: {config.llm_model_name}")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_config = AutoConfig.from_pretrained(config.llm_model_name)
    if getattr(llm_config, "tie_word_embeddings", None) is not False:
        llm_config.tie_word_embeddings = False
    llm_dtype = resolve_torch_dtype(config.llm_dtype, config.device)
    print(f"Using LLM dtype: {llm_dtype}")

    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_model_name,
        config=llm_config,
        torch_dtype=llm_dtype,
        device_map=config.device,
    )
    for param in llm.parameters():
        param.requires_grad = False
    llm.eval()
    if config.llm_hidden_size == 0:
        config.llm_hidden_size = llm.config.hidden_size
    print(f"LLM loaded. Hidden size: {config.llm_hidden_size}")

    # --- Load dataset ---
    print(f"Loading dataset: {config.dataset_path}")
    with open(config.dataset_path, "r") as f:
        all_records = json.load(f)

    # Filter valid records
    records = [
        r for r in all_records
        if isinstance(r.get("puzzle"), str) and len(r["puzzle"]) == 81
        and extract_nl_text(r).strip()
    ]
    print(f"Valid records: {len(records)} / {len(all_records)}")

    if len(records) == 0:
        raise ValueError(
            "No valid training records found after filtering. Expected each record to include "
            "a valid 81-char 'puzzle' and a non-empty text field such as 'nl_description' "
            "(allowed keys: nl_description/description/prompt/text/input)."
        )

    # Split into train / val
    if len(records) == 1:
        n_train, n_val = 1, 0
    else:
        n_val = max(1, int(len(records) * config.eval_split))
        n_val = min(n_val, len(records) - 1)
        n_train = len(records) - n_val
    generator = torch.Generator().manual_seed(config.seed)
    train_records, val_records = random_split(records, [n_train, n_val], generator=generator)
    train_records = [records[i] for i in train_records.indices]
    val_records = [records[i] for i in val_records.indices]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    train_dataset = SudokuNLDataset(train_records, tokenizer, config.llm_max_seq_len)
    val_dataset = SudokuNLDataset(val_records, tokenizer, config.llm_max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # --- Initialize bridge ---
    arch_cls = ARCH_REGISTRY[args.arch]
    bridge = arch_cls(config).to(config.device)
    n_params = count_parameters(bridge)
    print(f"Bridge architecture: {args.arch}")
    print(f"Trainable parameters: {n_params:,} ({n_params / 1e6:.2f}M)")
    assert n_params < 100_000_000, f"Bridge exceeds 100M param limit: {n_params:,}"

    use_wandb = args.wandb
    if use_wandb and wandb is None:
        raise ImportError("--wandb was set but package 'wandb' is not installed. Run: pip install wandb")

    if use_wandb:
        run_name = args.wandb_run_name or f"bridge_{args.arch}_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "arch": args.arch,
                "trainable_params": n_params,
                "llm": config.llm_model_name,
                "llm_layer_index": config.llm_layer_index,
                "dataset": config.dataset_path,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "patience": config.patience,
                "given_class_weight": config.given_class_weight,
                "transnar_recurrence_steps": config.transnar_recurrence_steps,
                "device": config.device,
            },
            settings=wandb.Settings(_disable_stats=True),
        )
        wandb.log({"num_params": n_params}, step=0)

    # --- Eval-only mode ---
    if args.eval_only:
        print(f"Loading checkpoint for eval: {args.eval_only}")
        ckpt_obj = load_checkpoint_any(args.eval_only, config.device)
        bridge.load_state_dict(extract_model_state_dict(ckpt_obj))
        metrics, pred_grids, target_grids = evaluate(bridge, llm, val_loader, config)
        print("\n=== Eval-only Results ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        errors = error_analysis(pred_grids, target_grids)
        print(f"  Error breakdown: {errors['error_counts']}")
        if use_wandb:
            wandb.log({f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=0)
            wandb.finish()
        return

    # --- Training ---
    optimizer = torch.optim.AdamW(
        bridge.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_fraction)
    scheduler = get_lr_schedule(optimizer, warmup_steps, total_steps)

    # Class-weighted loss: upweight digit classes to counteract empty-cell dominance
    class_weights = build_class_weights(config)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: empty=1.0, digits 1-9={config.given_class_weight}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    log_path = os.path.join(config.log_dir, f"translator_bridge_{args.arch}_{int(time.time())}.json")
    last_checkpoint_path = os.path.join(config.checkpoint_dir, f"last_{args.arch}.pt")

    training_log = {
        "config": {
            "arch": args.arch,
            "trainable_params": n_params,
            "llm": config.llm_model_name,
            "llm_layer_index": config.llm_layer_index,
            "dataset": config.dataset_path,
            "train_size": len(train_records),
            "val_size": len(val_records),
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "patience": config.patience,
            "given_class_weight": config.given_class_weight,
            "transnar_recurrence_steps": config.transnar_recurrence_steps,
        },
        "epochs": [],
    }

    best_exact_match = -1.0
    best_given_acc = -1.0
    best_checkpoint_path = None
    epochs_without_improvement = 0
    start_epoch = 0

    # --- Resume support (explicit path or automatic from last checkpoint) ---
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
        if not Path(resume_path).is_absolute():
            resume_path = str((repo_root / resume_path).resolve())
    elif not args.no_auto_resume and os.path.isfile(last_checkpoint_path):
        resume_path = last_checkpoint_path

    if resume_path and os.path.isfile(resume_path):
        print(f"Resuming training from: {resume_path}")
        resume_obj = load_checkpoint_any(resume_path, config.device)
        bridge.load_state_dict(extract_model_state_dict(resume_obj))

        if isinstance(resume_obj, dict) and "optimizer_state_dict" in resume_obj:
            optimizer.load_state_dict(resume_obj["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_obj:
                scheduler.load_state_dict(resume_obj["scheduler_state_dict"])
            start_epoch = int(resume_obj.get("epoch", 0))
            best_exact_match = float(resume_obj.get("best_exact_match", best_exact_match))
            best_given_acc = float(resume_obj.get("best_given_acc", best_given_acc))
            epochs_without_improvement = int(resume_obj.get("epochs_without_improvement", epochs_without_improvement))
            best_checkpoint_path = resume_obj.get("best_checkpoint_path", best_checkpoint_path)
            training_log = resume_obj.get("training_log", training_log)
            log_path = resume_obj.get("log_path", log_path)
            print(f"Resume state loaded: start_epoch={start_epoch}, best_exact={best_exact_match:.4f}, "
                  f"best_given={best_given_acc:.4f}, no_improve={epochs_without_improvement}")
        else:
            inferred_epoch = infer_epoch_from_checkpoint_path(resume_path)
            if inferred_epoch is not None:
                start_epoch = inferred_epoch
                print(
                    f"Checkpoint contains model weights only; starting optimizer/scheduler from scratch. "
                    f"Inferred start_epoch={start_epoch} from filename."
                )
            else:
                print("Checkpoint contains model weights only; starting optimizer/scheduler from scratch.")

        if config.epochs <= start_epoch:
            raise ValueError(
                f"Configured epochs ({config.epochs}) must be greater than resumed epoch ({start_epoch}). "
                f"Pass --epochs > {start_epoch} to continue training."
            )

    print(f"\nStarting training: {config.epochs} max epochs, early stopping patience={config.patience}")
    print(f"Total steps: {total_steps}, warmup: {warmup_steps} steps, then cosine decay (floor=10% peak LR)\n")

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_loss, train_acc, train_given_acc = train_one_epoch(
            bridge, llm, train_loader, optimizer, scheduler, criterion, config, epoch
        )
        val_metrics, pred_grids, target_grids = evaluate(bridge, llm, val_loader, config)
        elapsed = time.time() - t0

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_cell_accuracy": train_acc,
            "train_given_accuracy": train_given_acc,
            "val_loss": val_metrics["loss"],
            "val_cell_accuracy": val_metrics["cell_accuracy"],
            "val_exact_match": val_metrics["exact_match"],
            "val_given_accuracy": val_metrics["given_accuracy"],
            "val_empty_accuracy": val_metrics["empty_accuracy"],
            "val_format_valid_rate": val_metrics["format_valid_rate"],
            "elapsed_seconds": elapsed,
            "lr": scheduler.get_last_lr()[0],
        }
        training_log["epochs"].append(epoch_log)

        print(f"\nEpoch {epoch + 1}/{config.epochs} ({elapsed:.1f}s)")
        print(f"  Train: loss={train_loss:.4f}, cell_acc={train_acc:.4f}, given_acc={train_given_acc:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, cell_acc={val_metrics['cell_accuracy']:.4f}, "
              f"exact_match={val_metrics['exact_match']:.4f}, given_acc={val_metrics['given_accuracy']:.4f}, "
              f"empty_acc={val_metrics['empty_accuracy']:.4f}")

        # --- Early stopping on composite criterion ---
        # Primary: exact_match. Secondary (while exact_match is tied at 0): given_accuracy.
        # This prevents early stopping during the initial phase when exact_match is
        # stuck at 0 but given_accuracy is still climbing.
        improved = False
        if val_metrics["exact_match"] > best_exact_match:
            improved = True
        elif val_metrics["exact_match"] == best_exact_match and val_metrics["given_accuracy"] > best_given_acc:
            improved = True

        if improved:
            best_exact_match = val_metrics["exact_match"]
            best_given_acc = val_metrics["given_accuracy"]
            epochs_without_improvement = 0
            best_checkpoint_path = os.path.join(config.checkpoint_dir, f"best_{args.arch}.pt")
            torch.save(
                build_model_checkpoint_payload(
                    bridge,
                    config,
                    args.arch,
                    epoch=epoch + 1,
                    best_exact_match=best_exact_match,
                    best_given_acc=best_given_acc,
                ),
                best_checkpoint_path,
            )
            print(f"  ↑ New best: exact_match={best_exact_match:.4f}, given_acc={best_given_acc:.4f} → saved")
        else:
            epochs_without_improvement += 1
            print(f"  — No improvement ({epochs_without_improvement}/{config.patience})")

        # Periodic checkpoint
        if (epoch + 1) % config.save_every_epoch == 0 or (epoch + 1) == config.epochs:
            ckpt_path = os.path.join(config.checkpoint_dir, f"{args.arch}_epoch{epoch + 1}.pt")
            torch.save(
                build_model_checkpoint_payload(
                    bridge,
                    config,
                    args.arch,
                    epoch=epoch + 1,
                ),
                ckpt_path,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

        # Save log every epoch
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        # Full-state checkpoint for robust resume
        torch.save(
            {
                "arch": args.arch,
                "bridge_config": asdict(config),
                "epoch": epoch + 1,
                "model_state_dict": bridge.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_exact_match": best_exact_match,
                "best_given_acc": best_given_acc,
                "epochs_without_improvement": epochs_without_improvement,
                "best_checkpoint_path": best_checkpoint_path,
                "training_log": training_log,
                "log_path": log_path,
            },
            last_checkpoint_path,
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/cell_accuracy": train_acc,
                "train/given_accuracy": train_given_acc,
                "val/loss": val_metrics["loss"],
                "val/cell_accuracy": val_metrics["cell_accuracy"],
                "val/exact_match": val_metrics["exact_match"],
                "val/given_accuracy": val_metrics["given_accuracy"],
                "val/empty_accuracy": val_metrics["empty_accuracy"],
                "val/format_valid_rate": val_metrics["format_valid_rate"],
                "train/lr": scheduler.get_last_lr()[0],
                "train/epoch_seconds": elapsed,
            }, step=epoch + 1)

        # Early stopping check
        if epochs_without_improvement >= config.patience:
            print(f"\n⛔ Early stopping triggered after {epoch + 1} epochs "
                  f"(no improvement for {config.patience} epochs)")
            break

    # --- Final error analysis ---
    print("\n=== Final Error Analysis (validation set) ===")
    best_ckpt_obj = load_checkpoint_any(best_checkpoint_path, config.device)
    bridge.load_state_dict(extract_model_state_dict(best_ckpt_obj))
    val_metrics, pred_grids, target_grids = evaluate(bridge, llm, val_loader, config)
    errors = error_analysis(pred_grids, target_grids)

    stopped_epoch = len(training_log["epochs"])
    print(f"Stopped at epoch: {stopped_epoch}/{config.epochs}")
    print(f"Best val exact_match: {val_metrics['exact_match']:.4f}")
    print(f"Best val given_acc:   {val_metrics['given_accuracy']:.4f}")
    print(f"Best val empty_acc:   {val_metrics['empty_accuracy']:.4f}")
    print(f"Error breakdown: {errors['error_counts']}")
    print(f"Total cell errors: {errors['total_errors']}")

    training_log["final_metrics"] = val_metrics
    training_log["error_analysis"] = {
        "error_counts": errors["error_counts"],
        "total_errors": errors["total_errors"],
    }
    training_log["best_checkpoint"] = best_checkpoint_path
    training_log["stopped_epoch"] = stopped_epoch
    training_log["early_stopped"] = stopped_epoch < config.epochs

    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\nTraining log: {log_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")

    if use_wandb:
        wandb.summary["best_checkpoint"] = best_checkpoint_path
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                wandb.summary[f"final/{k}"] = v
        wandb.summary["final/total_errors"] = errors["total_errors"]
        wandb.finish()


if __name__ == "__main__":
    main()
