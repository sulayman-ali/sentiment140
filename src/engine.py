import torch
import torch.nn as nn
from tqdm import tqdm

def train(data_loader, model, optimizer, device):
    """
    function that trains model for one epoch
    :param data_loader: this is the torch dataloader :param model: model (lstm model)
    :param optimizer: torch optimizer, e.g. adam, sgd, etc.
    :param device: this can be "cuda" or "cpu"
    """
    
    # set model to training mode
    model.train()
    
    # iterate through each batch in data loader
    for data in tqdm(data_loader):
        # get tweets and targets
        tweets = data["tweet"]
        targets = data["target"]
        
        # move tensors to device
        tweets = tweets.to(device, dtype = torch.long) 
        targets = targets.to(device, dtype = torch.float)
        
        # zero gradients
        optimizer.zero_grad()
        
        # make predictions
        predictions = model(tweets)
        
        # calculate loss
        loss = nn.BCEWithLogitsLoss()(
            predictions,
            targets.view(-1,1)
        )
        
        # compute gradients
        loss.backward()
        
        # optimization step
        optimizer.step()
        
def evaluate(data_loader, model, device):
    final_predictions = []
    final_targets = []
    
    # change mode to eval
    model.eval()
    
    # disable gradient calculation
    with torch.no_grad():
        for data in data_loader:
            tweets = data['tweet']
            targets = data['target']
            tweets = tweets.to(device, dtype = torch.long)
            targets = targets.to(device, dtype = torch.float)
            
            # make predictions
            predictions = model(tweets)
            
            # move predictions + targets to list
            predictions = predictions.cpu().numpy().tolist()
            targets = data['target'].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)
            
    return final_predictions, final_targets