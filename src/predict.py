# code to take in sentence from command line argument 
# uses trained model to output prediction of positive or negative sentiment
import torch
import torch.nn as nn

import numpy as np
import pandas as pd 
import tensorflow as tf
import argparse
import pickle
from sklearn import metrics

import engine
import dataset
import config
import preprocessing    


def predict(sentence, model, tokenizer, device):
    """
    :param sentence: sentence to pass through model and recieve predictions on, string
    :
    """
    # preprocess sentence
    sentence = preprocessing.clean(sentence)
    
    # tokenize text
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])
    
    # zero pad text
    tokenized_sentence = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sentence, maxlen=config.MAX_LEN)
    
    # convert to torch tensor
    tokenized_sentence = torch.from_numpy(tokenized_sentence).to(device , dtype = torch.long)
    
    # put model in eval mode
    model.eval()
    
    with torch.no_grad():

        # make predictions
        prediction = model(tokenized_sentence)

        # move predictions to list
        prediction = prediction.cpu().numpy().tolist()
    
    output = np.array(prediction) >= 0.5
    
    if output == 1:
        sentiment = "POSITVE"
    else:
        sentiment = "NEGATIVE"
    
    return {'sentence':sentence, 'sentiment':sentiment}

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Output sentiment prediction for a tweet.')
    parser.add_argument('--text', type=str,required = True, help='text to classify')
    args = parser.parse_args()
    model = torch.load(config.MODEL_PATH)
    device = torch.device("cuda")

    with open('../input/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    print(predict(args.text,model, tokenizer, device))
