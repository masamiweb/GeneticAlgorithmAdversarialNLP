import os
import pandas as pd

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

