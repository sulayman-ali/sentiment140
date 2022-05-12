# code to evaluate trained model on test data
import torch
import torch.nn as nn

import numpy as np
import pandas as pd 
import tensorflow as tf
import pickle
from sklearn import metrics

import engine
import dataset
import config
import preprocessing    

if __name__ == "__main__":
    
    device = torch.device("cuda")

    # read test data
    df = pd.read_csv("../input/test.csv", header = None, encoding = config.DATASET_ENCODING)
    
    # assign column names
    df.columns = config.DATASET_COLUMNS
    
    # drop neutral sentiment
    df = df[df['target'] != 2]
    
    # map positive to 1, negative sentiment to 0
    df["sentiment"] = df.apply(
                lambda x: 1 if x['target'] == 4 else 0,
                axis = 1
                )
    
    # clean text
    df["text"] = df.apply(
                lambda x: preprocessing.clean(x['text']),
                axis = 1
                )
    # load tokenizer
    with open('../input/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # tokenize test data 
    xtest = tokenizer.texts_to_sequences(df.text.values)
    # zero pad 
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=config.MAX_LEN)

    # initialize test dataset class
    test_dataset = dataset.Sentiment140Dataset(
        tweets = xtest,
        targets = df.sentiment.values
    )
    
    # create torch dataloader for test
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers = 1
    )
    
    model = torch.load(config.MODEL_PATH)
    
    # evaluate model 
    outputs, targets = engine.evaluate(test_data_loader, model, device)

    # classification threshold is 0.5
    outputs = np.array(outputs) >= 0.5

    # calculate and output metrics
    class_report = metrics.classification_report(targets,outputs)
    
    print("Evaluating model: \n", class_report)