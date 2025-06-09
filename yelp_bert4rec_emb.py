import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

gpu_id = 1
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

full_seq_user = {}
with open("./data/yelp/seq_user_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        uid, seq = next(iter(obj.items()))
        full_seq_user[int(uid)] = seq

full_seq_item = {}
with open("./data/yelp/seq_item_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        iid, seq = next(iter(obj.items()))
        full_seq_item[int(iid)] = seq

with open("./data/yelp/trn_mat.pkl", "rb") as f:
    trn_mat = pickle.load(f).tocsr()
with open("./data/yelp/val_mat.pkl", "rb") as f:
    val_mat = pickle.load(f).tocsr()

merged_mat = (trn_mat + val_mat).astype(bool).astype(int).tocsr()

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

seq_user_data = make_split_user_seq(full_seq_user, merged_mat)
seq_item_data = make_split_item_seq(full_seq_item, merged_mat)

num_users = max(seq_user_data.keys()) + 1
num_items = max(seq_item_data.keys()) + 1
mask_token = num_items 
vocab_size = num_items + 1
max_len = 65
embed_dim = 256
batch_size = 256
epochs = 500
mask_prob = 0.15

class MaskedSeqDataset(Dataset):
    def __init__(self, seq_data):
        self.seqs = list(seq_data.values())

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx][-max_len:]
        if len(seq) < max_len:
            seq = [0] * (max_len - len(seq)) + seq
        inp = []
        tgt = []
        for x in seq:
            if x != 0 and np.random.rand() < mask_prob:
                inp.append(mask_token)
                tgt.append(x)
            else:
                inp.append(x)
                tgt.append(0)
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

class BERT4Rec(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8,
                                                   dim_feedforward=embed_dim*4,
                                                   dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.mlm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        pos = torch.arange(max_len, device=x.device).unsqueeze(0).expand_as(x)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        logits = self.mlm_head(h) 
        user_emb = h.mean(dim=1)
        return logits, user_emb

dataset = MaskedSeqDataset(seq_user_data)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = BERT4Rec().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=0)

best_loss = float('inf')
no_improve = 0
patience = 3

model.train()
for epoch in range(1, epochs + 1):
    total_loss = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        logits, _ = model(inp)
        loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f}")

    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"No improvement for {patience} epochs. Stopping early at epoch {epoch}.")
            break

model.eval()
with torch.no_grad():
    itm_emb_raw = model.token_emb.weight[:num_items].cpu().numpy()

def get_user_embeddings(seq_data):
    emb = np.zeros((num_users, embed_dim), dtype=np.float32)
    with torch.no_grad():
        for uid, seq in seq_data.items():
            seq_trim = seq[-max_len:]
            if len(seq_trim) < max_len:
                seq_trim = [0]*(max_len-len(seq_trim)) + seq_trim
            inp = torch.tensor([seq_trim], dtype=torch.long, device=device)
            _, uemb = model(inp)
            emb[uid] = uemb.squeeze(0).cpu().numpy()
    return emb

user_emb_raw = get_user_embeddings(seq_user_data)

with open("./data/yelp/usr_emb_bert4rec_raw.pkl", "wb") as f:
    pickle.dump(user_emb_raw, f)
with open("./data/yelp/itm_emb_bert4rec_raw.pkl", "wb") as f:
    pickle.dump(itm_emb_raw, f)

print("Saved BERT4Rec raw embeddings.")