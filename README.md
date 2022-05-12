# sentiment140

Repo contains code to train and evaluate an LSTM sentiment classifier trained on 1.6 milliion tweets from the sentiment 140 dataset

1. Unzip file in inputs directora and run `create_folds.py` to process dataset for training
2. run `train.py` to train, cross validate, and save classifier
3. run `eval.py` to compute performance metrics on test dataset
