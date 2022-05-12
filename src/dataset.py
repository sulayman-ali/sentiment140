import torch
# simple dataset class to return a sample of training / validation data 

class Sentiment140Dataset:
    def __init__(self,tweets,targets):
        """
        :param tweets: numpy array representing tweets
        :param targets: vector, a numpy array also
        """
        self.tweets = tweets
        self.targets = targets
        
    def __len__(self):
        # return length of dataset
        return len(self.tweets)
    
    def __getitem__(self,item):
        # given item index (integer), return tweet and target as torch tensor
        tweet = self.tweets[item, :]
        target = self.targets[item]
        
        return {
            "tweet": torch.tensor(tweet, dtype = torch.long),
            "target": torch.tensor(target, dtype = torch.float)
        }