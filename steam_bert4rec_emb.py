import os
import json
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 0) 설정 (item2vec과 동일한 형태)
# =========================================================
DATA_DIR = "./data/steam"
SEQ_USER_PATH = os.path.join(DATA_DIR, "seq_user_data.jsonl")
SEQ_ITEM_PATH = os.path.join(DATA_DIR, "seq_item_data.jsonl")
TRN_MAT_PATH = os.path.join(DATA_DIR, "trn_mat.pkl")
VAL_MAT_PATH = os.path.join(DATA_DIR, "val_mat.pkl")

OUT_USER_PKL = os.path.join(DATA_DIR, "usr_emb_bert4rec_raw.pkl")
OUT_ITEM_PKL = os.path.join(DATA_DIR, "itm_emb_bert4rec_raw.pkl")

GPU_ID = 0
SEED = 2026

# BERT4Rec hyperparams
MAX_LEN = 50
EMBED_DIM = 1024
BATCH_SIZE = 128
EPOCHS = 500
MASK_PROB = 0.1
LR = 5e-5

N_HEADS = 8
NUM_LAYERS = 4
FF_MULT = 4
DROPOUT = 0.3

PATIENCE = 3
NUM_WORKERS = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 1) Utils
# =========================================================
def read_seq_jsonl(path: str):
    seqs = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            k, v = next(iter(obj.items()))
            seqs[int(k)] = v
    return seqs


def make_split_user_seq(full_seq_user, mat):
    split_seq_user = {}
    for uid, seq in full_seq_user.items():
        allowed_items = set(mat[uid].indices)
        split_seq_user[uid] = [i for i in seq if i in allowed_items]
    return split_seq_user


def make_split_item_seq(full_seq_item, mat):
    split_seq_item = {}
    for iid, seq in full_seq_item.items():
        allowed_users = set(mat[:, iid].nonzero()[0])
        split_seq_item[iid] = [u for u in seq if u in allowed_users]
    return split_seq_item


def shift_user_item_ids(seq_user_dict, offset=1):
    # Shift item ids so that 0 is reserved for padding.
    return {uid: [i + offset for i in seq] for uid, seq in seq_user_dict.items()}


# =========================================================
# 2) Dataset
# =========================================================
class MaskedSeqDataset(Dataset):
    def __init__(self, seq_data, max_len, mask_prob, mask_token):
        self.seqs = list(seq_data.values())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx][-self.max_len :]
        if len(seq) < self.max_len:
            seq = [0] * (self.max_len - len(seq)) + seq

        inp = []
        tgt = []
        for x in seq:
            if x != 0 and np.random.rand() < self.mask_prob:
                inp.append(self.mask_token)
                tgt.append(x)
            else:
                inp.append(x)
                tgt.append(0)
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# =========================================================
# 3) Model
# =========================================================
class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, max_len, embed_dim, n_heads, num_layers, ff_mult, dropout):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        pad_mask = (x == 0)
        pos = torch.arange(self.max_len, device=x.device).unsqueeze(0).expand_as(x)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        logits = self.mlm_head(h)
        # Mean pooling over non-padding tokens
        valid = (~pad_mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp(min=1.0)
        user_emb = (h * valid).sum(dim=1) / denom
        return logits, user_emb


# =========================================================
# 4) Main
# =========================================================

def main():
    seed_everything(SEED)
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    full_seq_user = read_seq_jsonl(SEQ_USER_PATH)
    full_seq_item = read_seq_jsonl(SEQ_ITEM_PATH)

    with open(TRN_MAT_PATH, "rb") as f:
        trn_mat = pickle.load(f).tocsr()
    with open(VAL_MAT_PATH, "rb") as f:
        val_mat = pickle.load(f).tocsr()

    merged_mat = (trn_mat + val_mat).astype(bool).astype(int).tocsr()

    seq_user_data = make_split_user_seq(full_seq_user, merged_mat)
    seq_item_data = make_split_item_seq(full_seq_item, merged_mat)
    # Reserve 0 for padding; shift item ids to 1..num_items
    seq_user_data = shift_user_item_ids(seq_user_data, offset=1)

    num_users = max(seq_user_data.keys()) + 1
    num_items = max(seq_item_data.keys()) + 1
    print(f"[INFO] #users={num_users}, #items={num_items}")

    # vocab: [PAD=0] + items(1..num_items) + [MASK=num_items+1]
    mask_token = num_items + 1
    vocab_size = num_items + 2

    dataset = MaskedSeqDataset(
        seq_user_data,
        max_len=MAX_LEN,
        mask_prob=MASK_PROB,
        mask_token=mask_token,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    model = BERT4Rec(
        vocab_size=vocab_size,
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        num_layers=NUM_LAYERS,
        ff_mult=FF_MULT,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float("inf")
    no_improve = 0

    print("[INFO] Train BERT4Rec (MLM)")
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits, _ = model(inp)
            loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"[EPOCH {epoch}/{EPOCHS}] avg_loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[INFO] Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print("[INFO] Extract embeddings...")
    model.eval()
    with torch.no_grad():
        # item id i maps to token id (i + 1)
        itm_emb_raw = model.token_emb.weight[1:num_items + 1].cpu().numpy().astype(np.float32)

    def get_user_embeddings(seq_data):
        emb = np.zeros((num_users, EMBED_DIM), dtype=np.float32)
        with torch.no_grad():
            for uid, seq in seq_data.items():
                if not seq:
                    continue
                seq_trim = seq[-MAX_LEN:]
                if len(seq_trim) < MAX_LEN:
                    seq_trim = [0] * (MAX_LEN - len(seq_trim)) + seq_trim
                inp = torch.tensor([seq_trim], dtype=torch.long, device=device)
                _, uemb = model(inp)
                emb[uid] = uemb.squeeze(0).cpu().numpy()
        return emb

    user_emb_raw = get_user_embeddings(seq_user_data)

    with open(OUT_USER_PKL, "wb") as f:
        pickle.dump(user_emb_raw, f)
    with open(OUT_ITEM_PKL, "wb") as f:
        pickle.dump(itm_emb_raw, f)

    print("[SAVE]", OUT_USER_PKL)
    print("[SAVE]", OUT_ITEM_PKL)
    print("[DONE] Steam BERT4Rec embedding generation finished.")


if __name__ == "__main__":
    main()
