# Non-LLMRec: A Lightweight Recommendation Framework Inspired by LLM Analysis

## Data download
To run the code, please download the required dataset (zip file):
```base
cd data/
gdown --id YOUR_FILE_ID --output data.zip
unzip data.zip

## Examples to run the codes

The command to evaluate the backbone models and RLMRec is as follows.

- **Backbone**
  ```bash
  python encoder/train_encoder.py --model {model_name} --dataset {dataset} --cuda 0

- **RLMRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_plus --dataset {dataset} --cuda 0

- **Non-LLMRec**
  ```bash
  python encoder/train_encoder.py --model {model_name}_plus --dataset {dataset} --emb bert4rec_knn --cuda 0

Supported models/datasets:
- **model_name:** `gccf`, `lightgcn`, `sgl`, `simgcl`, `autocf`
- **dataset:** `amazon`, `yelp`
