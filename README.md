# sentiment140

Repo contains code to train and evaluate an LSTM sentiment classifier trained on 1.6 milliion tweets from the sentiment 140 dataset

1. Unzip dataset file (`sentiment140.zip`) in `input` directory and run `create_folds.py` to process dataset for training
2. Download and expand fast text wiki news vectors 300d https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip. Store in `input` directory
3. run `train.py` to train, cross validate, and save classifier
4. run `eval.py` to compute performance metrics on test dataset


-- 

To recieve a prediction on a single sentence
- run `predict.py --text "i love pytorch"`
