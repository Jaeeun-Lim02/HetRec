# Revisiting LLM-Enhanced Collaborative Filtering: A Text-Free Alternative with Heterogeneous Interaction Signals

Official PyTorch implementation of **HetRec**, accepted as a short paper at
**CIKM 2026**.

HetRec revisits auxiliary representation alignment for collaborative filtering
and asks whether its benefits require LLM-generated knowledge. Instead of using
text or LLM inference, HetRec constructs complementary user and item
representations from heterogeneous interaction views, enriches them with
similarity-aware k-nearest-neighbor aggregation, and aligns them with a base CF
model through contrastive learning.

**Paper:** [ACM Digital Library / DOI](https://doi.org/10.1145/3799682.3839906)
(the link will become active after publication)

<!-- Export Figure 2 from the final paper to assets/hetrec_overview.png. -->
<p align="center">
  <img src="assets/hetrec_overview.png" width="850" alt="Overview of HetRec">
</p>

## Environment

The code is intended to run on Linux with an NVIDIA GPU. The following setup is
recommended for reproducibility:

- Linux
- Python 3.9
- PyTorch 1.13.1
- CUDA 11.6
- PyTorch Geometric extensions: `torch-scatter` and `torch-sparse`

Other Python 3.7+ and PyTorch versions may work, but have not been tested.

```bash
conda create -n hetrec python=3.9 -y
conda activate hetrec

pip install torch==1.13.1+cu116 \
  torchvision==0.14.1+cu116 \
  torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-1.13.1+cu116.html

pip install numpy scipy scikit-learn pyyaml tqdm gdown
```

If a different CUDA version is used, install the matching PyTorch and PyTorch
Geometric wheels.

## Data

Download and extract the preprocessed datasets as follows:

```bash
cd data
gdown "https://drive.google.com/uc?id=1_dn4S6z-y64pOluQ2vcselaSmNrkW8do"
unzip data.zip
mv data/* .
rmdir data
rm data.zip
cd ..
```

Each dataset directory follows this structure:

```text
amazon/  # The same structure is used for yelp and steam.
├── trn_mat.pkl                   # Training interactions
├── val_mat.pkl                   # Validation interactions
├── tst_mat.pkl                   # Test interactions
├── usr_prf.pkl                   # User profile text for the RLMRec baseline
├── itm_prf.pkl                   # Item profile text for the RLMRec baseline
├── usr_emb_np.pkl                # User text embedding for the RLMRec baseline
├── itm_emb_np.pkl                # Item text embedding for the RLMRec baseline
├── usr_emb_bert4rec_knn.pkl      # User BERT4Rec + kNN embedding for HetRec
├── itm_emb_bert4rec_knn.pkl      # Item BERT4Rec + kNN embedding for HetRec
├── usr_emb_item2vec_knn.pkl      # User Item2Vec + kNN embedding for HetRec
└── itm_emb_item2vec_knn.pkl      # Item Item2Vec + kNN embedding for HetRec
```

The interaction-derived auxiliary embeddings are precomputed offline. Running
HetRec therefore requires neither text input nor an LLM API.

## Usage

Replace `<model>` and `<dataset>` with one of the supported names listed below.

### Backbone

```bash
python encoder/train_encoder.py \
  --model <model> \
  --dataset <dataset> \
  --cuda 0
```

### RLMRec baseline

```bash
python encoder/train_encoder.py \
  --model <model>_plus \
  --dataset <dataset> \
  --cuda 0
```

### HetRec with BERT4Rec auxiliary embeddings

```bash
python encoder/train_encoder.py \
  --model <model>_plus \
  --dataset <dataset> \
  --emb bert4rec_knn \
  --cuda 0
```

### HetRec with Item2Vec auxiliary embeddings

```bash
python encoder/train_encoder.py \
  --model <model>_plus \
  --dataset <dataset> \
  --emb item2vec_knn \
  --cuda 0
```

Supported options:

- `<model>`: `sgl`, `simgcl`, `gccf`, `autocf`, or `lightgcn`
- `<dataset>`: `amazon`, `yelp`, or `steam`

For example:

```bash
python encoder/train_encoder.py \
  --model sgl_plus \
  --dataset amazon \
  --emb bert4rec_knn \
  --cuda 0
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{lim2026revisiting,
  author    = {Lim, Jaeeun and Lee, Jae-woong},
  title     = {Revisiting {LLM}-Enhanced Collaborative Filtering: A Text-Free Alternative with Heterogeneous Interaction Signals},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year      = {2026},
  publisher = {Association for Computing Machinery},
  location  = {Rome, Italy},
  doi       = {10.1145/3799682.3839906},
  url       = {https://doi.org/10.1145/3799682.3839906}
}
```

Publication metadata such as page numbers can be added after the ACM Digital
Library record becomes available.

## License

This project is released under the [Apache License 2.0](LICENSE). Please also
comply with the licenses and terms of the upstream codebases and datasets used
in this project.
