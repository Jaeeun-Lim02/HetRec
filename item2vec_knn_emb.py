import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 0) 설정 (item2vec과 동일한 형태)
# =========================================================
DATASET = "amazon"
DATA_DIR = os.path.join("./data", DATASET)

IN_USER_PKL = os.path.join(DATA_DIR, "usr_emb_item2vec_raw.pkl")
IN_ITEM_PKL = os.path.join(DATA_DIR, "itm_emb_item2vec_raw.pkl")

TRN_MAT_PATH = os.path.join(DATA_DIR, "trn_mat.pkl")
VAL_MAT_PATH = os.path.join(DATA_DIR, "val_mat.pkl")

OUT_USER_PKL = os.path.join(DATA_DIR, "usr_emb_item2vec_knn.pkl")
OUT_ITEM_PKL = os.path.join(DATA_DIR, "itm_emb_item2vec_knn.pkl")

TOP_K = 10
ALPHA = 0.8


# =========================================================
# 1) Utils
# =========================================================

def upgrade(emb, sim, top_k, alpha):
    out = emb.copy()
    for i in range(sim.shape[0]):
        idx = np.argsort(-sim[i])
        idx = idx[idx != i][:top_k]
        w = sim[i, idx]
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / top_k
        knn_avg = np.average(emb[idx], axis=0, weights=w)
        out[i] = (1 - alpha) * emb[i] + alpha * knn_avg
    return out


# =========================================================
# 2) Main
# =========================================================

def main():
    print(f"[INFO] dataset={DATASET}, top_k={TOP_K}, alpha={ALPHA}")

    with open(IN_USER_PKL, "rb") as f:
        user_emb_raw = pickle.load(f)
    with open(IN_ITEM_PKL, "rb") as f:
        itm_emb_raw = pickle.load(f)

    with open(TRN_MAT_PATH, "rb") as f:
        trn_mat = pickle.load(f).tocsr()
    with open(VAL_MAT_PATH, "rb") as f:
        val_mat = pickle.load(f).tocsr()

    merged_mat = (trn_mat + val_mat).astype(bool).astype(int).tocsr()

    user_emb_knn = upgrade(user_emb_raw, cosine_similarity(merged_mat), top_k=TOP_K, alpha=ALPHA)
    item_emb_knn = upgrade(itm_emb_raw, cosine_similarity(merged_mat.T), top_k=TOP_K, alpha=ALPHA)

    with open(OUT_USER_PKL, "wb") as f:
        pickle.dump(user_emb_knn, f)
    with open(OUT_ITEM_PKL, "wb") as f:
        pickle.dump(item_emb_knn, f)

    print("[SAVE]", OUT_USER_PKL)
    print("[SAVE]", OUT_ITEM_PKL)
    print("[DONE] Item2Vec KNN embedding generation finished.")


if __name__ == "__main__":
    main()
