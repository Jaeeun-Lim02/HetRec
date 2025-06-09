import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

dataset = "amazon"

with open(f"./data/{dataset}/usr_emb_bert4rec_raw.pkl", "rb") as f:
    user_emb_raw = pickle.load(f)
with open(f"./data/{dataset}/itm_emb_bert4rec_raw.pkl", "rb") as f:
    itm_emb_raw = pickle.load(f)

with open(f"./data/{dataset}/trn_mat.pkl", "rb") as f:
    trn_mat = pickle.load(f).tocsr()
with open(f"./data/{dataset}/val_mat.pkl", "rb") as f:
    val_mat = pickle.load(f).tocsr()
merged_mat = (trn_mat + val_mat).astype(bool).astype(int).tocsr()

def upgrade(emb, sim, top_k=10, alpha=0.8):
    out = emb.copy()
    for i in range(sim.shape[0]):
        idx = np.argsort(-sim[i])
        idx = idx[idx != i][:top_k]
        w = sim[i, idx]
        w = w / w.sum() if w.sum()>0 else np.ones_like(w)/top_k
        knn_avg = np.average(emb[idx], axis=0, weights=w)
        out[i] = (1 - alpha) * emb[i] + alpha * knn_avg
    return out

user_emb_knn = upgrade(user_emb_raw, cosine_similarity(merged_mat), top_k=10, alpha=0.8)
item_emb_knn = upgrade(itm_emb_raw, cosine_similarity(merged_mat.T), top_k=10, alpha=0.8)

with open(f"./data/{dataset}/usr_emb_bert4rec_knn.pkl", "wb") as f:
    pickle.dump(user_emb_knn, f)
with open(f"./data/{dataset}/itm_emb_bert4rec_knn.pkl", "wb") as f:
    pickle.dump(item_emb_knn, f)

print("Saved KNN-augmented embeddings.")