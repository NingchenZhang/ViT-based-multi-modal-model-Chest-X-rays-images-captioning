import pandas as pd
import os
import sys
sys.path.extend(["../", "../../", "../../../"])

from matplotlib.ticker import FormatStrFormatter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.vocab import GloVe
import json


image_folder = 'textmodel/images_training'
label_folder = 'processed_data.csv'
df = pd.read_csv(label_folder)
random_state = 420 #0 #define the random seed
def creat_tokenier(df):
    col = ['image_1', 'image_2', 'impression', 'xml file name']
    df = df[col].copy()
    # path
    df['image_1'] = df['image_1'].apply(
        lambda row: os.path.join(image_folder, row))
    df['image_2'] = df['image_2'].apply(lambda row: os.path.join(image_folder, row))

    df['impression_final'] = '<CLS> ' + df.impression + ' <END>'
    #Bulid the input Sequence and output sequence
    df['impression_ip'] = '<CLS> ' + df.impression
    df['impression_op'] = df.impression + ' <END>'
    df['impression'].value_counts()
    df.drop_duplicates(subset=['xml file name'], inplace=True)

    k = df['impression'].value_counts()
    df = df.merge(k,
                  left_on='impression',
                  right_index=True)  # join left impression value with right index
    #change the column name
    # df.columns = ['impression', 'image_1', 'image_2', 'impression_x', 'xml file name', 'impression_final',
    #               'impression_ip', 'impression_op', 'impression_counts']
    #df.columns = ['image_1', 'image_2', 'impression', 'xml file name', 'impression_final',
     #             'impression_ip', 'impression_op', 'impression_counts']
    df.columns = ['impression_x', 'image_1', 'image_2', 'impression', 'xml file name', 'impression_final',
                   'impression_ip', 'impression_op', 'impression_counts']
    # del df['impression_x']  # deleting impression_x column
    return df
def splite_data(df):
    #only get 10% from counts which lower than 5
    other1 = df[df['impression_counts'] > 5]  # selecting those datapoints which have impression valuecounts >5
    other2 = df[df['impression_counts'] <= 5]  # selecting those datapoints which have impression valuecounts <=5
    train, test = train_test_split(other1, stratify=other1['impression'].values, test_size=0.1, random_state=random_state)
    # inorder to promise the data balance
    # Q1
    test_other2_sample = other2.sample(int(0.2 * other2.shape[0]),
                                       random_state=random_state)  # getting some datapoints from other2 data for test data
    other2 = other2.drop(test_other2_sample.index, axis=0)
    # here i will be choosing 0.5 as the test size as to create a reasonable size of test data
    test = pd.concat([test, test_other2_sample])
    test_data = test.reset_index(drop=True)

    train = pd.concat([train, other2])
    train = train.reset_index(drop=True)
    print(train.shape[0],test.shape[0])
    # To using the upsample and downsample to balance all the data.

    df_majority = train[train['impression_counts'] >= 100]  # having value counts >=100
    df_minority = train[train['impression_counts'] <= 5]  # having value counts <=5
    df_other = train[
        (train['impression_counts'] > 5) & (train['impression_counts'] < 100)]  # value counts between 5 and 100
    n1 = df_minority.shape[0]
    n2 = df_majority.shape[0]
    n3 = df_other.shape[0]
    # upsample them to 30

    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=3 * n1,
                                     random_state=random_state)
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=n2 // 15,
                                       random_state=random_state)
    df_other_downsampled = resample(df_other,
                                    replace=False,
                                    n_samples=n3 // 10,
                                    random_state=random_state)

    train = pd.concat([df_majority_downsampled, df_minority_upsampled, df_other_downsampled])
    # train = pd.concat([df_minority, df_majority, df_other])

    train_data = train.reset_index(drop=True)

    return train_data,test_data



class Tokenizer:
    def __init__(self, train_data, test_data):
        # self.tokenizer = get_tokenizer('spacy')

        self.train_data = train_data
        self.test_data = test_data
        self.glove = GloVe(name='6B', dim=300)
        # Divide sentences into words
        # self.train_captions = [self.tokenizer(x) for x in train_data.impression_final.values]
        # self.test_captions = [self.tokenizer(x) for x in test_data.impression_final.values]
        # self.train_captions = [ ['<cls>']+self.tokenizer(x)+ ['<end>'] for x in train_data.impression.values]
        # self.test_captions = [ ['<cls>']+self.tokenizer(x)+['<end>'] for x in test_data.impression.values]
        self.train_captions = [ ['begin']+x.split(' ')+['end'] for x in train_data.impression.values]
        self.test_captions = [ ['begin']+x.split(' ')+['end'] for x in test_data.impression.values]

        # Build vocab
        token_counter = Counter()
        for tokens in self.train_captions+self.test_captions:
            token_counter.update(tokens)
       
        specials = ['<unk>', '<pad>']

        for special in specials:
            token_counter[special] = 1

        min_freq = 1
        filtered_token_counter = {token: count for token, count in token_counter.items() if count >= min_freq}

        #Create vocabulary file of this IU datasets
        # token_to_index = sorted(list(filtered_token_counter.keys()), key=lambda x: filtered_token_counter[x], reverse=True)
        # token_to_index = {item: idx for idx, item in enumerate(token_to_index)}
        #just load the vocabulary list because we have finish create via 132,133 two rows code.
        token_to_index = json.load(open('token_to_index.json'))

        self.vocab = Vocab(token_to_index)
        self.itos = {index: token for index, token in enumerate(token_to_index.keys())}

        self.vocab_size = len(self.vocab)

        # Get the start and end index
        self.start_index = self.vocab['begin']
        self.end_index = self.vocab['end']
        self.unk_index = self.vocab['<unk>']  # Get the unk index
        self.pad_index = self.vocab['<pad>']  # Get the pad index

        # Convert texts to sequences
        self.train_sequences = []
        self.test_sequences = []
        for caption in self.train_captions:
            sequence = [self.vocab[token] if token in self.vocab else self.unk_index for token in caption]
            # sequence = self.glove.get_vecs_by_tokens(caption, True)
            self.train_sequences.append(torch.tensor(sequence))
        for caption in self.test_captions:
            sequence = [self.vocab[token] if token in self.vocab else self.unk_index for token in caption]
            self.test_sequences.append(torch.tensor(sequence))

        # Compute the caption lengths
        self.caption_len = [len(x) for x in self.train_sequences]

        # print(
        #     'The 80 percentile value of caption_len which is %i will be taken as the maximum padded value for each impression' % (
        #         np.percentile(self.caption_len, 80)))
        print(
            'The 80 percentile value of caption_len which is %i will be taken as the maximum padded value for each impression' % (
                np.percentile(self.caption_len, 80)))
        # self.max_pad = int(np.percentile(self.caption_len, 80))
        self.max_pad = int(np.percentile(self.caption_len, 80))


    def tokenize_and_pad(self, text, prefix=None):
        # sequence = [self.vocab[token] if token in self.vocab else self.unk_index for token in self.tokenizer(text)]
        sequence = [str(token) if token in self.vocab else self.unk_index for token in text.split(' ')]#self.tokenizer(text)]

        if prefix == 'cls':
            sequence = ['begin'] + sequence
            sequence = [item for item in sequence if item != '.']
        elif prefix == 'end':
            sequence = sequence + ['end']
            sequence = [item for item in sequence if item != '.']

        if len(sequence) > self.max_pad:
            sequence = sequence[:self.max_pad]
        else:
            # sequence += [self.pad_index] * (self.max_pad - len(sequence))
            sequence += ['<pad>'] * (self.max_pad - len(sequence)) # Use the pad index
        return sequence
        # return torch.tensor(sequence)

    def get_vocab_size(self):
        print("get_vocab_size: ", len(self.vocab))
        return len(self.vocab)

# Please remember to create a new Tokenizer object to apply these changes.

def plot_and_save_metrics(train_loss, train_acc, val_loss, val_acc, save_path="metrics.png"):
    epochs = range(1, len(train_loss) + 1)

    formatter = FormatStrFormatter('%.4f')

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

