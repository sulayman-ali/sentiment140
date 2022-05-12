import torch
import torch.nn as nn

# our LSTM model 

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        """
        :param embedding_matrix: numpy array w/ vectors for all words
        """
        
        super(LSTM, self).__init__()
        # number of words is the number of rows in our embedding matrix
        num_words = embedding_matrix.shape[0]
        
        # dimension of our embeddings is the number of columns in the matrix
        embed_dim = embedding_matrix.shape[1]
        
        # input embedding layer
        self.embedding = nn.Embedding(
            num_embeddings = num_words,
            embedding_dim = embed_dim
        )
        
        # set weights of embedding layer equal to embedding matrix
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype = torch.float32
            )
        )
        
        # freeze pretrained embeddings
        self.embedding.weight.requres_grad = False
        
        # BiLSTM w/ 128 hidden size
        self.lstm = nn.LSTM(
            embed_dim,
            128,
            bidirectional = True,
            batch_first = True
        )
        
        # linear output layer
        # single output node
        # input is 512 b/c mean and max pooling end up concatenated 
        # for each direction (BiLSTM) | (128*2)*2
        self.out = nn.Linear(512, 1)
        
    def forward(self, x):
        # embed tokens
        x = self.embedding(x)
        
        # pass embedding output to lstm
        x, _ = self.lstm(x)
        
        # apply mean and max pooling on lstm output
        avg_pool = torch.mean(x, 1)
        max_pool = torch.max(x, 1)[0]
        
        # concatenate mean and max pooling
        out = torch.cat((avg_pool, max_pool),1)
        
        # pass through linear layer
        out = self.out(out)
        
        return out