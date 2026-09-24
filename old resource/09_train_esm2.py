#!/usr/bin/env python3
"""
09_train_esm2.py
================
Trains the ESM2 Attention architecture on exactly 250 plasmids and 250 chromosomes
(using seed 42) and tests it on the benchmark_1000 dataset.
"""

import os
import json
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR = BASE_DIR / "data" / "raw" / "benchmark_1000"
RESULTS_DIR = BASE_DIR / "results" / "pretrained_baselines"

ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# --- Common Utilities ---
def load_and_subsample(fasta_path, n_samples, seed):
    if not fasta_path.exists():
        raise FileNotFoundError(f"Missing file: {fasta_path}")
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    rng = np.random.default_rng(seed)
    if len(records) > n_samples:
        indices = rng.choice(len(records), size=n_samples, replace=False)
        records = [records[i] for i in indices]
    return [str(r.seq).upper() for r in records]

def extract_orfs(dna_seq, min_len=100, max_orfs=10):
    seq_obj = Seq(dna_seq)
    orfs = []
    for strand, seq in [(1, seq_obj), (-1, seq_obj.reverse_complement())]:
        for frame in range(3):
            try:
                l = len(seq) - frame
                padded = seq[frame:frame + (l - l % 3)]
                peps = str(padded.translate()).split("*")
                orfs.extend([p for p in peps if len(p) >= min_len])
            except:
                continue
    # Sort by length descending and take the top `max_orfs`
    orfs.sort(key=len, reverse=True)
    return orfs[:max_orfs]

# --- Feature Extraction (Unpooled) ---
def get_unpooled_esm_features(seqs, device, desc=""):
    print(f"\nLoading {ESM_MODEL_NAME} for Feature Extraction...")
    tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM_MODEL_NAME).to(device)
    model.eval()
    
    X_raw = []
    for seq in tqdm(seqs, desc=f"ESM-2 Extraction {desc}"):
        orfs = extract_orfs(seq, min_len=100)
        if not orfs:
            X_raw.append(torch.zeros(1, model.config.hidden_size))
            continue
            
        embeddings = []
        batch_size = 8
        for i in range(0, len(orfs), batch_size):
            batch = orfs[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_emb = torch.sum(last_hidden * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                embeddings.append((sum_emb / sum_mask).cpu())
                
        seq_emb_unpooled = torch.cat(embeddings, dim=0) # Shape: (num_orfs, 320)
        X_raw.append(seq_emb_unpooled)
        
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return X_raw

# --- PyTorch Attention Model ---
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x) # (seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=0)
        context = torch.sum(attn_weights * x, dim=0, keepdim=True) # (1, hidden_size)
        return context

class EsmAttentionClassifier(nn.Module):
    def __init__(self, hidden_size=320):
        super().__init__()
        self.attn_pool = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        context = self.attn_pool(x) # (1, hidden_size)
        logits = self.classifier(context) # (1, 1)
        return logits.squeeze()

# --- Dataset ---
class UnpooledDataset(Dataset):
    def __init__(self, X_list, y_array):
        self.X = X_list
        self.y = torch.tensor(y_array, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def collate_fn(batch):
    X = [item[0] for item in batch]
    y = torch.stack([item[1] for item in batch])
    return X, y

def main():
    seed = 42
    n_train_samples = 250
    epochs = 15
    lr = 1e-3

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Training Data (250 Plasmids, 250 Chromosomes)
    print("\n[1/3] Loading Training Data...")
    train_plasmid_path = TRAIN_DIR / "train_plasmid.fasta"
    train_chromosome_path = TRAIN_DIR / "train_chromosome.fasta"
    
    train_plas_seqs = load_and_subsample(train_plasmid_path, n_train_samples, seed)
    train_chrom_seqs = load_and_subsample(train_chromosome_path, n_train_samples, seed)
    X_train_raw = get_unpooled_esm_features(train_plas_seqs + train_chrom_seqs, device, desc="Train")
    y_train = np.array([1]*len(train_plas_seqs) + [0]*len(train_chrom_seqs))

    train_dataset = UnpooledDataset(X_train_raw, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # 2. Train Model
    print("\n[2/3] Training ESM-2 Attention Model...")
    model = EsmAttentionClassifier(hidden_size=320).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0_train = time.time()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = []
            for x in batch_X:
                x = x.to(device)
                logit = model(x)
                logits.append(logit)
            logits = torch.stack(logits)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f}")
    t1_train = time.time()

    # 3. Evaluate on Benchmark_1000
    print("\n[3/3] Evaluating on Benchmark_1000...")
    test_plasmid_path = TEST_DIR / "test_plasmid_1000.fasta"
    test_chromosome_path = TEST_DIR / "test_chromosome_1000.fasta"
    
    test_plas_seqs = load_and_subsample(test_plasmid_path, 1000, seed) # Load all available
    test_chrom_seqs = load_and_subsample(test_chromosome_path, 1000, seed)
    
    X_test_raw = get_unpooled_esm_features(test_plas_seqs + test_chrom_seqs, device, desc="Test")
    y_test = np.array([1]*len(test_plas_seqs) + [0]*len(test_chrom_seqs))
    
    test_dataset = UnpooledDataset(X_test_raw, y_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model.eval()
    y_preds = []
    y_trues = []
    
    t0_infer = time.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            for x in batch_X:
                x = x.to(device)
                logit = model(x)
                prob = torch.sigmoid(logit).item()
                y_preds.append(1 if prob > 0.5 else 0)
            y_trues.extend(batch_y.numpy())
    t1_infer = time.time()

    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, zero_division=0)
    prec = precision_score(y_trues, y_preds, zero_division=0)
    rec = recall_score(y_trues, y_preds, zero_division=0)
    cm = confusion_matrix(y_trues, y_preds)

    print("\n=== ESM-2 Architecture Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    
    metrics = {
        "classifier": "ESM2 Attention",
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "train_time_s": t1_train - t0_train,
        "infer_time_s": t1_infer - t0_infer
    }
    
    out_json = RESULTS_DIR / "esm2_results.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved ESM2 metrics to {out_json}")

if __name__ == "__main__":
    main()
