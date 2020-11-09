import os
import pandas as pd

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
    """
    Need to redo this to deal with the IMDB dataset - rename columns to match the sentiments140 dataset so we can 
    re-use all th e below methods without changing anything
    """

    
    pos_train = file_path + "train/pos"
    pos_train_files = [pos_train + '/' + x for x in os.listdir(pos_train) if x.endswith('.txt')]
   
    pos_test = file_path + "test/pos"
    pos_test_files = [pos_test + '/' + x for x in os.listdir(pos_test) if x.endswith('.txt')]
    
    # list containing the file path to all positive sentiment text
    all_pos = pos_train_files + pos_test_files
    
    
    neg_train = file_path + "train/neg"
    neg_train_files = [neg_train + '/' + x for x in os.listdir(neg_train) if x.endswith('.txt')]
    
    neg_test = file_path + "test/neg"
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
    
#     print("neg shape: ", df_neg.shape)
#     print("pos shape: ", df_pos.shape)
    frames = [df_neg, df_pos]
    data_file = pd.concat(frames, axis=0)
    
    #df4['column3'] = np.where(df4['gender'] == '', df4['name'], df4['gender'])
    
    return data_file 

