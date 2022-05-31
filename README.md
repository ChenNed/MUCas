# MUCas
This is a TensorFlow implementation of paper "Multi-Scale Graph Capsule with Influence Attention for Information Cascades Size Prediction", will come soon.
# Overview
- `data/` put the download dataset here;
- `model/` contains the implementation of the MUCas;
- `utils/` contains preprocessing code：
    * preprocess the original Weibo dataset (`preprocess_weibo.py`);
    * split the data to train set, validation set and test set (`split_dataset.py`);
    * trainsform the datasets to the format of ".pkl" (`gen_model_input.py`).
