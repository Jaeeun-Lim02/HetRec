# Temporal and Neighborhood Heterogeneity for Collaborative Filtering Inspired by LLM-based Models

## Data download
To run the code, please download the required dataset (zip file):
```base
cd data/
# pip install gdown
gdown --id 1kWnAfqCF8MS9FIyVvrpJg0cHOkFysNBu --output data.zip

unzip data.zip
mv data/amazon . && mv data/yelp . && rm -rf data
```

Each dataset consists of:
```
amazon(yelp/steam)
├── trn_mat.pkl      # training set (sparse matrix)
├── val_mat.pkl      # validation set (sparse matrix)
├── tst_mat.pkl      # test set (sparse matrix)
├── usr_prf.pkl      # user profile text
├── itm_prf.pkl      # item profile text
├── usr_emb_np.pkl   # user text embedding
├── itm_emb_np.pkl   # item text embedding
├── usr_emb_bert4rec_knn.pkl   # user bert4rec + knn embedding
├── itm_emb_bert4rec_knn.pkl   # item bert4rec + knn embedding
├── usr_emb_item2vec_knn.pkl   # user item2vec + knn embedding
└── itm_emb_item2vec_knn.pkl   # item item2vec + knn embedding
```

## Examples to run the codes

The command to evaluate the backbone models and RLMRec is as follows.

- **Backbone**
  ```bash
  python encoder/train_encoder.py --model {model_name} --dataset {dataset} --cuda 0

- **RLMRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_plus --dataset {dataset} --cuda 0

- **HetRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_plus --dataset {dataset} --emb bert4rec_knn --cuda 0
  python encoder/train_encoder.py --model {model_name}_plus --dataset {dataset} --emb item2vec_knn --cuda 0

Supported models/datasets:
- **model_name:** `sgl`, `simgcl`, `gccf`, `autocf`, `lightgcn`
- **dataset:** `amazon`, `yelp`
