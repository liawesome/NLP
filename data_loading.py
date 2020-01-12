# Dataset Prepartion
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import data_path, test_data_path, dev_data_path, train_data_path
import json
import nltk
import sklearn
import pandas as pd
import string

# load the rawdata and keep the first 4000 rows
def constructTrainMatrix():
    col_names=['index','position','content']
    train= pd.read_csv('/Users/zhali/Documents/TD interview/train.txt',encoding = "ISO-8859-1", low_memory=False,header=None,names=col_names)
    labels= pd.read_csv('/Users/zhali/Documents/TD interview/labels_Candidate.csv', header=None,names=['sentiment'])
    #to group the content within the same index
    train_df = train.groupby('index').agg(lambda x: list(x)).drop(columns=['position'])
    train_df.describe()  #EDA of the dataset
    train_df["labels"] = labels
    train_df = train_df[:3999]
    return train_df

# keep the unknown test data
def constructTestMatrix():
    col_names=['index','position','content']
    train= pd.read_csv('/Users/zhali/Documents/TD interview/train.txt',encoding = "ISO-8859-1", low_memory=False,header=None,names=col_names)
    labels= pd.read_csv('/Users/zhali/Documents/TD interview/labels_Candidate.csv', header=None,names=['sentiment'])
    #to group the content within the same index
    train_df = train.groupby('index').agg(lambda x: list(x)).drop(columns=['position'])
    train_df = train_df[3999:]
    train_df["labels"] = -1 #undefined
    return train_df


# create data object to put each comment's label, target and index into a big matrix
def new_load_raw_data(matrix, data = {'file_names':[], 'target':[], 'data':[]}):
    for index, row in matrix.iterrows():
        data['file_names'].append(index)
        data['target'].append(int(row['labels']))
        data['data'].append(' '.join(row['content']))
    return data

# create output file
def write_processed_data(data, outfile_path):
    with open(outfile_path, 'w') as outfile:
        json.dump(data, outfile)


def init():

    write_processed_data(new_load_raw_data(constructTrainMatrix(),data = {'file_names':[], 'target':[], 'data':[]}),train_data_path)
    write_processed_data(new_load_raw_data(constructTestMatrix(),data = {'file_names':[], 'target':[], 'data':[]}),test_data_path)

def load_data(path):
    return json.load(open(path, 'r'))


#print(new_load_raw_data(constructMatrix()))

# EXECUTE init() only if data/processed_data/ folder is empty
init()
