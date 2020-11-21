import os
import pandas as pd
import string 
import re
from . import normalize_dataset as nd

'''
Large movie review database for binary sentiment classification
This is a dataset for binary sentiment classification containing substantially more data than previous 
benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 
There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. 
See the README file contained in the release for more details.
Download from:
https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

See this paper for more information:
https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf
'''

def process_dataset_IMDB(file_path: str):
    
    print("Creating dataset, please wait ...")
    pos_train = file_path + "/train/pos"
    pos_train_files = [pos_train + '/' + x for x in os.listdir(pos_train) if x.endswith('.txt')]
   
    pos_test = file_path + "/test/pos"
    pos_test_files = [pos_test + '/' + x for x in os.listdir(pos_test) if x.endswith('.txt')]
    
    # list containing the file path to all positive sentiment text
    all_pos = pos_train_files + pos_test_files
    
    
    neg_train = file_path + "/train/neg"
    neg_train_files = [neg_train + '/' + x for x in os.listdir(neg_train) if x.endswith('.txt')]
    
    neg_test = file_path + "/test/neg"
    neg_test_files = [neg_test + '/' + x for x in os.listdir(neg_test) if x.endswith('.txt')]
    
    # list containing the file path to all negative sentiment text
    all_neg = neg_train_files + neg_test_files
    
    df_pos = pd.DataFrame(columns = ['sentiment', 'text'])
    for i, l in enumerate(all_pos):
            f = open(all_pos[i])
            line = f.readline()
            df_pos = df_pos.append({'sentiment': 1, 'text': line}, ignore_index=True)
            f.close()
    
    df_neg = pd.DataFrame(columns = ['sentiment', 'text'])
    for i, l in enumerate(all_neg):
            f = open(all_neg[i])
            line = f.readline()
            df_neg = df_neg.append({'sentiment': 0, 'text': line}, ignore_index=True)
            f.close()
    
    # add the two data files into one dataframe
    frames = [df_neg, df_pos]
    df = pd.concat(frames, axis=0)
    
    #set column data types
    df['sentiment'] = df['sentiment'].astype(str).astype(int)
    df['text'] = df['text'].astype(str)
    
    df = df.sample(frac = 1, random_state = 7) 
    
    # there is no need for normalizing, it is done just to be able to display wordcloud and other graphs cleanly
    # normalizing is done in the model as a callback used by TextVectorization 
    print("Normalizing dataset, please wait...")
    df = nd.clean_and_return(df, 'text')
    
    
    print("Dataset created!\n")
    
    return df 

def process_dataset_Sentiment140(file_path: str ):
    
    print("Creating dataset, please wait ...")
    # load the data file into a data frame
    df = pd.read_csv(file_path, encoding='latin-1', header=None) # changed encoding to 'latin-1'
    
    # Rename the columns so we can reference them later
    df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
    
    # drop all the columns we don't need
    df = df.drop(['id', 'date', 'query', 'user_id'], axis=1) 
    
    # change all 4's to 1's (just for neatness)
    df.loc[df['sentiment'] == 4, 'sentiment'] = 1
    
    df['sentiment'] = df['sentiment'].astype(str).astype(int)
    df['text'] = df['text'].astype(str)
    
    # there is no need for normalizing, it is done just to be able to display wordcloud and other graphs cleanly
    # normalizing is done in the model as a callback used by TextVectorization 
    print("Normalizing dataset, please wait...")
    df = nd.clean_and_return(df,'text')
    
    df = df.sample(frac = 1, random_state = 7) 
    print("Dataset created!\n")
    return df


