import json
import re
import html
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, models

# ----------------------- Utilities -----------------------
def decode_html_entities(text):
    return html.unescape(text)

def preprocess_text(text):
    # HTML 엔티티 디코딩 → 소문자 변환 → 특수문자 제거
    text = decode_html_entities(text).lower()
    return re.sub(r'[^a-z0-9\s]', '', text)

def load_json_lines(filepath):
    objs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return objs

# ----------------------- Load data -----------------------
data_dir     = './data/steam'
user_file    = f'{data_dir}/steam_user_prompts.json'
item_file    = f'{data_dir}/steam_item_prompts.json'

user_data = load_json_lines(user_file)
item_data = load_json_lines(item_file)

# ------------------- Model init -----------------------
# 1) Transformer
word_embedding_model = models.Transformer(
    model_name_or_path='sentence-transformers/all-MiniLM-L6-v2',
    max_seq_length=512
)
# 2) mean+max pooling
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_max_tokens=True
)
# 3) 조합
model = SentenceTransformer(modules=[word_embedding_model, pooling_model],
                            device='cuda:1')

# ------------------- Embed prompts --------------------
# 1) 원본 prompt 텍스트 리스트 추출 + 전처리
user_prompts = [ preprocess_text(u['prompt']) for u in user_data ]
item_prompts = [ preprocess_text(i['prompt']) for i in item_data ]

# 2) 배치 단위로 임베딩 (numpy array 반환)
user_embs = model.encode(user_prompts, convert_to_numpy=True, batch_size=64)
item_embs = model.encode(item_prompts, convert_to_numpy=True, batch_size=64)

# ------------------- Save to .pkl ----------------------
with open(f'{data_dir}/usr_emb_pr_np.pkl', 'wb') as f:
    pickle.dump(user_embs, f)
with open(f'{data_dir}/itm_emb_pr_np.pkl', 'wb') as f:
    pickle.dump(item_embs, f)

print("Saved user embeddings to usr_emb_pr_np.pkl (shape:", user_embs.shape, ")")
print("Saved item embeddings to itm_emb_pr_np.pkl (shape:", item_embs.shape, ")")