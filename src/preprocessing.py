import re

def clean(text):
    """
    :param text: expects string holding tweet text to be cleaned
    """
    USER = '@[\w_]+'
    LINK = 'https?:\/\/\S+'
    HASHTAG = '#\S+'
    NUMBER = '\d+'
    PUNCTUATIONS = '[\.?!,;:\-\[\]\{\}\(\)\'\"/]'
    
    # replace @ handles 
    user_sub = re.sub(USER, ' <user> ', text)
    # replace urls 
    link_sub = re.sub(LINK, ' <url> ', user_sub)
    # replace hashtags
    hashtag_sub = re.sub(HASHTAG, ' <hashtag> ', link_sub)
    # replace numbers
    number_sub = re.sub(NUMBER, ' <number> ', hashtag_sub)
    # remove punctuation
    clean_text = re.sub(PUNCTUATIONS, ' ', number_sub)

    return clean_text.lower()