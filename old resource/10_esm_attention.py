#!/usr/bin/env python3
"""
10_esm_attention.py
===================
ESM-2 Prototype with Trainable Attention Mechanism.
Replaces simple Mean Pooling with a neural attention layer to focus on critical ORFs.
"""

import argparse
import time
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

DATA_DIR = Path("./data/raw")
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# --- Common Utilities ---
def load_and_subsample(fasta_path, n_samples, seed):
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    rng = np.random.default_rng(seed)
    if len(records) > n_samples:
        indices = rng.choice(len(records), size=n_samples, replace=False)
        records = [records[i] for i in indices]
    return [str(r.seq).upper() for r in records]

def extract_orfs(dna_seq, min_len=100):
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
    return orfs

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
        # x shape: (seq_len, hidden_size)
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
        # x is (seq_len, hidden_size) representing unpooled ORFs for a single DNA sequence
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
    # We process batch size 1 conceptually, but we can return list
    X = [item[0] for item in batch]
    y = torch.stack([item[1] for item in batch])
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50, help="Number of sequences per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    plasmid_path = DATA_DIR / "test_plasmid.fasta"
    chromosome_path = DATA_DIR / "test_chromosome.fasta"
    print(f"\nLoading {args.n_samples} samples per class...")
    plas_seqs = load_and_subsample(plasmid_path, args.n_samples, args.seed)
    chrom_seqs = load_and_subsample(chromosome_path, args.n_samples, args.seed)
    all_seqs = plas_seqs + chrom_seqs
    y = np.array([1]*len(plas_seqs) + [0]*len(chrom_seqs))

    # Extract Unpooled Features
    X_raw = get_unpooled_esm_features(all_seqs, device, desc="All Sequences")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, stratify=y, random_state=args.seed)
    
    train_dataset = UnpooledDataset(X_train, y_train)
    test_dataset = UnpooledDataset(X_test, y_test)
    
    # Batch size 1 because sequences have different lengths (num_orfs)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Initialize Model
    model = EsmAttentionClassifier(hidden_size=320).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print("\nTraining Attention Model...")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            
            # Forward pass (one by one due to variable length)
            # Alternatively, process the whole batch iteratively
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
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    print("\nEvaluating Model...")
    model.eval()
    y_preds = []
    y_trues = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            for x in batch_X:
                x = x.to(device)
                logit = model(x)
                prob = torch.sigmoid(logit).item()
                y_preds.append(1 if prob > 0.5 else 0)
            y_trues.extend(batch_y.numpy())

    print("\n=== ESM-2 Attention Prototype Results ===")
    print(classification_report(y_trues, y_preds, target_names=["Chromosome", "Plasmid"]))
    print(f"Accuracy: {accuracy_score(y_trues, y_preds):.4f}")
    print(f"F1-score: {f1_score(y_trues, y_preds):.4f}")

if __name__ == "__main__":
    main()
