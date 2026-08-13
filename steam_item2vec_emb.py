import os
import json
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


# =========================================================
# 0) 설정 (그냥 실행되도록 하드코딩)
# =========================================================
DATA_DIR = "./data/steam"
SEQ_USER_PATH = os.path.join(DATA_DIR, "seq_user_data.jsonl")
SEQ_ITEM_PATH = os.path.join(DATA_DIR, "seq_item_data.jsonl")
TRN_MAT_PATH = os.path.join(DATA_DIR, "trn_mat.pkl")
VAL_MAT_PATH = os.path.join(DATA_DIR, "val_mat.pkl")

# ✅ output: 기존 파일 덮어쓰기 싫으면 이름 바꾸면 됨
OUT_USER_PKL = os.path.join(DATA_DIR, "usr_emb_item2vec_raw.pkl")
OUT_ITEM_PKL = os.path.join(DATA_DIR, "itm_emb_item2vec_raw.pkl")

GPU_ID = 0
SEED = 2026

# Item2Vec hyperparams
EMBED_DIM = 256
WINDOW = 5
NEG_K = 10
EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 4096

# user embedding extraction
USER_LAST_K = 20
USER_POOLING = "recent_linear"   # "mean" or "recent_linear"

# negative sampling
NEG_POWER = 0.75  # 0.5 / 0.75 / 1.0 튜닝 가능

# ✅ Dynamic window / Subsampling (핵심 개선)
USE_DYNAMIC_WINDOW = True
USE_SUBSAMPLING = False
SUBSAMPLE_T = 5e-4     # 1e-5 ~ 1e-4 추천

# ✅ item embedding 저장 방식
SAVE_INOUT_AVG = True   # True면 (in+out)/2 저장, False면 in_emb만 저장

# (옵션) 저장 시 L2 normalize (side embedding에서 종종 유리)
L2_NORMALIZE = True

# for speed
NUM_WORKERS = 2


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_unigram_table(seqs, num_items, power=0.75):
    counts = np.zeros(num_items, dtype=np.float64)
    for s in seqs:
        for x in s:
            if 0 <= x < num_items:
                counts[x] += 1.0
    probs = counts ** power
    probs = probs / (probs.sum() + 1e-12)
    return probs.astype(np.float64), counts


def build_subsample_keep_prob(counts, t=1e-5):
    """
    Word2Vec subsampling keep probability:
    P_keep(w) = min(1, (sqrt(f/t) + 1) * (t/f))
    """
    total = counts.sum() + 1e-12
    f = counts / total
    keep = np.ones_like(f, dtype=np.float64)
    nz = f > 0
    keep[nz] = (np.sqrt(f[nz] / t) + 1.0) * (t / f[nz])
    keep = np.clip(keep, 0.0, 1.0)
    return keep.astype(np.float64)


def l2_normalize(x: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


# =========================================================
# 2) Iterable Dataset: (center, context) pairs
# =========================================================
class SkipGramPairDataset(IterableDataset):
    def __init__(self, user_seqs, window=5, use_dynamic_window=True,
                 use_subsampling=False, keep_prob=None):
        super().__init__()
        self.user_seqs = user_seqs
        self.window = window
        self.use_dynamic_window = use_dynamic_window
        self.use_subsampling = use_subsampling
        self.keep_prob = keep_prob  # np.array[num_items]

    def __iter__(self):
        for seq in self.user_seqs:
            if len(seq) < 2:
                continue

            # ✅ Subsampling: 인기 아이템을 확률적으로 drop
            if self.use_subsampling and self.keep_prob is not None:
                filtered = []
                for x in seq:
                    if random.random() < float(self.keep_prob[x]):
                        filtered.append(x)
                seq = filtered
                if len(seq) < 2:
                    continue

            L = len(seq)
            for i in range(L):
                c = seq[i]

                # ✅ Dynamic window
                if self.use_dynamic_window:
                    w = random.randint(1, self.window)
                else:
                    w = self.window

                left = max(0, i - w)
                right = min(L, i + w + 1)

                for j in range(left, right):
                    if j == i:
                        continue
                    ctx = seq[j]
                    yield c, ctx


def collate_pairs(batch):
    centers = torch.tensor([b[0] for b in batch], dtype=torch.long)
    contexts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return centers, contexts


# =========================================================
# 3) Item2Vec Model (SGNS)
# =========================================================
class Item2Vec(nn.Module):
    def __init__(self, num_items, embed_dim):
        super().__init__()
        self.in_emb = nn.Embedding(num_items, embed_dim)
        self.out_emb = nn.Embedding(num_items, embed_dim)

        nn.init.normal_(self.in_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.out_emb.weight, mean=0.0, std=0.02)

    def forward(self, center, pos_ctx, neg_ctx):
        v = self.in_emb(center)               # (B,D)
        u_pos = self.out_emb(pos_ctx)         # (B,D)
        u_neg = self.out_emb(neg_ctx)         # (B,K,D)

        pos_score = (v * u_pos).sum(dim=1)    # (B,)
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # (B,K)
        return pos_score, neg_score


# =========================================================
# 4) Main
# =========================================================
def main():
    seed_everything(SEED)
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    # --- Load 4 files (same protocol: train+val merged)
    full_seq_user = read_seq_jsonl(SEQ_USER_PATH)
    full_seq_item = read_seq_jsonl(SEQ_ITEM_PATH)

    with open(TRN_MAT_PATH, "rb") as f:
        trn_mat = pickle.load(f).tocsr()
    with open(VAL_MAT_PATH, "rb") as f:
        val_mat = pickle.load(f).tocsr()

    merged_mat = (trn_mat + val_mat).astype(bool).astype(int).tocsr()

    seq_user_data = make_split_user_seq(full_seq_user, merged_mat)
    seq_item_data = make_split_item_seq(full_seq_item, merged_mat)

    num_users = max(seq_user_data.keys()) + 1
    num_items = max(seq_item_data.keys()) + 1
    print(f"[INFO] #users={num_users}, #items={num_items}")

    # --- sequences list for training
    user_seqs = list(seq_user_data.values())

    # --- negative sampling distribution + counts
    neg_probs, counts = build_unigram_table(user_seqs, num_items=num_items, power=NEG_POWER)

    # --- subsampling keep prob
    keep_prob = None
    if USE_SUBSAMPLING:
        keep_prob = build_subsample_keep_prob(counts, t=SUBSAMPLE_T)
        print(f"[INFO] Subsampling enabled: t={SUBSAMPLE_T}")

    # --- dataset / loader
    dataset = SkipGramPairDataset(
        user_seqs,
        window=WINDOW,
        use_dynamic_window=USE_DYNAMIC_WINDOW,
        use_subsampling=USE_SUBSAMPLING,
        keep_prob=keep_prob,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_pairs,
    )

    # --- model
    model = Item2Vec(num_items=num_items, embed_dim=EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    # --- training
    print("[INFO] Train Item2Vec (SGNS)")
    model.train()

    rng = np.random.default_rng(SEED)

    for ep in range(1, EPOCHS + 1):
        total_loss = 0.0
        steps = 0

        for centers, pos_ctx in loader:
            centers = centers.to(device)
            pos_ctx = pos_ctx.to(device)

            # negative sampling (CPU->GPU)
            neg_ctx = rng.choice(
                num_items,
                size=(centers.size(0), NEG_K),
                replace=True,
                p=neg_probs,
            )
            neg_ctx = torch.tensor(neg_ctx, dtype=torch.long, device=device)

            pos_score, neg_score = model(centers, pos_ctx, neg_ctx)

            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)

            loss_pos = bce(pos_score, pos_label)
            loss_neg = bce(neg_score, neg_label)
            loss = loss_pos + loss_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if steps % 200 == 0:
                print(f"[EP {ep}/{EPOCHS}] step={steps}, loss={total_loss/steps:.6f}")

        print(f"[EPOCH {ep}/{EPOCHS}] avg_loss={total_loss/max(steps,1):.6f}")

    # --- extract item embeddings
    print("[INFO] Extract embeddings...")
    model.eval()
    with torch.no_grad():
        in_emb = model.in_emb.weight.detach().cpu().numpy().astype(np.float32)
        out_emb = model.out_emb.weight.detach().cpu().numpy().astype(np.float32)

    if SAVE_INOUT_AVG:
        itm_emb = (in_emb + out_emb) / 2.0
        print("[INFO] item embedding = (in_emb + out_emb)/2")
    else:
        itm_emb = in_emb
        print("[INFO] item embedding = in_emb only")

    if L2_NORMALIZE:
        itm_emb = l2_normalize(itm_emb)
        print("[INFO] L2-normalized item embeddings")

    # --- user embedding
    usr_emb = np.zeros((num_users, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for uid, seq in seq_user_data.items():
            if len(seq) == 0:
                continue
            s = seq[-USER_LAST_K:] if USER_LAST_K > 0 else seq
            vecs = itm_emb[np.array(s, dtype=np.int64)]

            if USER_POOLING == "mean":
                usr_emb[uid] = vecs.mean(axis=0)
            elif USER_POOLING == "recent_linear":
                # ✅ 최근일수록 더 큰 가중치
                w = np.linspace(1.0, 2.0, num=len(s), dtype=np.float32)
                w = w / (w.sum() + 1e-12)
                usr_emb[uid] = (vecs * w[:, None]).sum(axis=0)
            else:
                usr_emb[uid] = vecs.mean(axis=0)

    # --- save
    with open(OUT_USER_PKL, "wb") as f:
        pickle.dump(usr_emb, f)
    with open(OUT_ITEM_PKL, "wb") as f:
        pickle.dump(itm_emb, f)

    print("[SAVE]", OUT_USER_PKL)
    print("[SAVE]", OUT_ITEM_PKL)
    print("[DONE] Steam Item2Vec enhanced embedding generation finished.")


if __name__ == "__main__":
    main()
