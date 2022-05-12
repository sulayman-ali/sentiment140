import pandas as pd
import preprocessing
import config
from sklearn import model_selection

if __name__ == "__main__":

    # read training data
    df = pd.read_csv("../input/train.csv", header = None, encoding = config.DATASET_ENCODING)
    
    # assign column names
    df.columns = config.DATASET_COLUMNS
    
    # map positive to 1, negative sentiment to 0, neutral sentiment to 2
    df["sentiment"] = df.apply(
                lambda x: 1 if x['target'] == 4 else 0,
                axis = 1
                )
    
    # clean text
    df["text"] = df.apply(
                lambda x: preprocessing.clean(x['text']),
                axis = 1
                )
    
    # create kfolds
    df["kfold"] = -1
    
    # randomize dataframe
    df = df.sample(frac = 1).reset_index(drop = True)
    
    # get labels
    y = df.sentiment.values
    
    # instantiate kfold class
    kf = model_selection.KFold(n_splits = 5)
    
    # fill kfold colun in dataframe
    for fold, (t_, v_) in enumerate(kf.split(X=df,y=y)):
        df.loc[v_, 'kfold'] = fold
        
    df.to_csv("../input/sentiment_140_folds.csv", index = False)