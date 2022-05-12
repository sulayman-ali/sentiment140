import io
import torch

import numpy as np
import pandas as pd 
import pickle
import tensorflow as tf

from sklearn import metrics 
from tqdm import tqdm

import config
import dataset
import engine
import lstm

def load_vectors(fname):
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
            fname,
            'r',
            encoding='utf-8',
            newline='\n', 
            errors='ignore'
    )

    n, d = map(int, fin.readline().split()) 
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:])) 
            
    return data

def create_embedding_matrix(word_index, embedding_dict):
    """
    This function creates the embedding matrix.
    :param word_index: a dictionary with word:index_value
    :param embedding_dict: a dictionary with word:embedding_vector 
    :return: a numpy array with embedding vectors for all known words
    """
    # initialize matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    # loop over all the words
    for word, i in tqdm(word_index.items()):
        # if word is found in pre-trained embeddings, 
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    # return embedding matrix
    return embedding_matrix

def run(df, fold, cv = True):
    """
    Run training and validation for a given fold and dataset
    :param df: pandas dataframe with kfold column
    :param fold: current fold, int
    :param cv: flag for running cross validation, 
    """
    
    # fetch training dataframe . filter based on fold
    train_df = df[df.kfold != fold].reset_index(drop=True)
    
    # fetch validation dataframe. filter based on fold 
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    
    # fit tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer() 
    tokenizer.fit_on_texts(df.text.values.tolist())
    
    # save tokenizer
    print("Saving tokenizer")
    with open('../input/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # tokenize training data
    xtrain = tokenizer.texts_to_sequences(train_df.text.values)
    # tokenize validation data 
    xtest = tokenizer.texts_to_sequences(valid_df.text.values)
    
    # zero pad the training sequences given the maximum length
    # if sequence is > MAX_LEN, it is truncated on left hand side too 
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=config.MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=config.MAX_LEN)

    
    # initialize training dataset class
    train_dataset = dataset.Sentiment140Dataset(
        tweets = xtrain,
        targets = train_df.sentiment.values
    )
    
    # create dataloader
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers = 2
    )
    
    # initialize validation dataset class
    valid_dataset = dataset.Sentiment140Dataset(
        tweets = xtest,
        targets = valid_df.sentiment.values
    )
    
    # create torch dataloader for validation
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers = 1
    )
    
    print("Loading embeddings")
    # load embeddings
    #embedding_dict = load_vectors("../input/crawl-300d-2M.vec")
    embedding_dict = load_vectors("../input/wiki-news-300d-1M-subword.vec")
    print("Creating embedding matrix")
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict)
    
    # create torch device
    device = torch.device("cuda")
    
    # get model
    model = lstm.LSTM(embedding_matrix)
    
    # send model to device
    model.to(device)
    
    # adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    print("Training Model")
    # set best accuracy to zero
    best_accuracy = 0 
    # set early stopping counter to zero
    early_stopping_counter = 0
    # train and validate 
    for epoch in tqdm(range(config.EPOCHS)):
        
        # train one epoch
        engine.train(train_data_loader, model, optimizer, device)
        
        # evaluate model 
        outputs, targets = engine.evaluate(valid_data_loader, model, device)
        
        # classification threshold is 0.5
        outputs = np.array(outputs) >= 0.5
        
        # calculate accuracy
        accuracy = metrics.accuracy_score(targets,outputs)
        f1 = metrics.f1_score(targets, outputs)
     
        print(
        f"FOLD:{fold}, Epoch: {epoch}, Accuracy Score = {accuracy}, F1 Score = {f1}"
        )


        # simple early stopping
        if accuracy > best_accuracy: 
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter > 2: 
            break
            
    return model 
    
if __name__ == "__main__":
    # load data
    df = pd.read_csv("../input/sentiment_140_folds.csv")
    # train for all folds
    # run(df, fold=0) 
    # run(df, fold=1) 
    # run(df, fold=2) 
    # run(df, fold=3) 
    net = run(df, fold=0)
    model_path = config.MODEL_PATH
    # save
    torch.save(net, model_path)
