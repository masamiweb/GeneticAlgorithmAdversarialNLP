import string 
import re
import pandas as pd

"""
preprocess-twitter.py

Script for preprocessing tweets by Romain Paulus

Translation of Ruby script to create features for GloVe vectors for Twitter data.
http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

this version from gist.github.com/ppope > preprocess_twitter.py

Extended by Manjinder Singh, to enable removal of html entities, and punctuation symbols (except apostrophe)
"""
FLAGS = re.MULTILINE | re.DOTALL



# additional html entities added to remove (Manjinder Singh)
html_entities = [" quot ", " amp ", " lt ", " gt ", " circ ", " tilde ", " ensp ", " emsp ", " thinsp ", " zwnj ", " zwj ", 
                     " lrm ", " rlm ", " ndash ", " mdash ", " lsquo ", " rsquo ", " sbquo ", " ldquo ", " rdquo ", " bdquo ", " permil ", " lsaquo ", " rsaquo "]

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return  result

def allcaps(text):
    text = text.group()
    return text.lower() # removed tag


def tokenize(text):
    
    punctuation = string.punctuation
    punctuation = punctuation.translate({ord(i):None for i in "'"}) # keep the apostrophe
       
    # lower case all text
    text = text.lower()
    # remove extra spaces so we can then remove all amp and quot chars correctly, also removes trailing spaces
    text = ' '.join(text.split()) 
    
    
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "") # url
    text = re_sub(r"@\w+", "") # twitter username
    text = re_sub(r"&\w+", "") # remove html entities starting with a &
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "") # smile
    text = re_sub(r"{}{}p+".format(eyes, nose), "") # lolface
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "") # sadface
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "") # neutralface
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","") # heart
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "") # remove numbers
    text = re_sub(r"#\w+", "")  # remove hashtag
    text = re_sub(r"([!?.]){2,}", r"\1 ") # remove punctuation repetitions eg. "!!!" 
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 ") # remove elongated words and trim eg. shorten 'Awwwwwwwww' to 'Aw'
    

    ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    #text = re_sub(r"([A-Z]){2,}", allcaps)  # moved below -amackcrane

    # amackcrane additions
    text = re_sub(r"([a-zA-Z<>()])([?!.:;,])", r"\1 \2")
    text = re_sub(r"\(([a-zA-Z<>]+)\)", r"( \1 )")
    text = re_sub(r"  ", r" ")
    #text = re_sub(r" ([A-Z]){2,} ", allcaps) # lowercase all caps
    
    
    # finally remove all punctuation and numbers
    text  = "".join([char for char in text if char not in punctuation])
    text = re.sub('[0-9]+', '', text)
    
    # remove all html entities
    for h in html_entities:
        if h in text:
            text = re_sub(h, "")
    

    return text.lower()

cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
def cleanhtml(raw_html):
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def clean_and_return(data_frame, text_col_name: str):
    print("Cleaning dataset, please wait ...")
    data_frame[text_col_name] = data_frame[text_col_name].apply(lambda x: cleanhtml(x))
    data_frame[text_col_name] = data_frame[text_col_name].apply(lambda x: tokenize(x))
    print("Dataset cleaned!")
    return data_frame