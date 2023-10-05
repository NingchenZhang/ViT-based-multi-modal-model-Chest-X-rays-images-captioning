import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import pandas as pd
from utils import Tokenizer
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe

image_folder = 'textmodel/images_training'
label_folder = 'processed_data.csv'
df = pd.read_csv(label_folder)
ViT_weights = 'weight/model-45.pth'

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
    df.columns = ['impression', 'image_1', 'image_2', 'impression_x', 'xml file name', 'impression_final',
                  'impression_ip', 'impression_op', 'impression_counts']
    del df['impression_x']  # deleting impression_x column
    return df

class MyDataset(Dataset):
    def __init__(self, df, input_size, text_processor, augmentation=True):
        self.image1 = df.image_1
        self.image2 = df.image_2
        # self.caption = df.impression_ip #inout
        # self.caption1 = df.impression_op#output
        self.caption = df.impression #inout
        self.caption1 = df.impression#output

        self.input_size = input_size
        self.text_processor = text_processor
        self.augmentation = augmentation

        self.glove = GloVe(name='6B', dim=300)
        # image enhancement
        # Horizontal flip
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)

        # Vertical flip
        self.vertical_flip = transforms.RandomVerticalFlip(p=1)

    def __getitem__(self, i):
        image1 = cv2.imread(self.image1[i], cv2.IMREAD_UNCHANGED) / 255
        image2 = cv2.imread(self.image2[i], cv2.IMREAD_UNCHANGED) / 255
        image1 = cv2.resize(image1, self.input_size, interpolation = cv2.INTER_NEAREST)
        image2 = cv2.resize(image2, self.input_size, interpolation = cv2.INTER_NEAREST)

        # Convert images to PyTorch tensors
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

        # Convert texts to sequences and pad them
        caption = self.text_processor.tokenize_and_pad(self.caption[i], 'cls')
        # caption = self.glove.get_vecs_by_tokens(caption, True)
        caption = torch.tensor([self.text_processor.vocab[token] if token in self.text_processor.vocab else self.unk_index for token in caption])
 
        caption1 = self.text_processor.tokenize_and_pad(self.caption1[i], 'end')#'cls')
        caption1 = torch.tensor([self.text_processor.vocab[token] if token in self.text_processor.vocab else self.unk_index for token in caption1])
        # print('caption shape: ',caption.shape)
        # print('caption1 shape: ', caption1.shape)

        # Apply augmentations image enhancement
        if self.augmentation:
            a = np.random.uniform()
            if a < 0.333:
                image1 = self.horizontal_flip(image1)
                image2 = self.horizontal_flip(image2)
            elif a < 0.667:
                image1 = self.vertical_flip(image1)
                image2 = self.vertical_flip(image2)
            else:
                pass
        return image1, image2, caption, caption1

    def __len__(self):
        return len(self.image1)

#-------------just for the test-------------
if __name__ == '__main__':
    dataframe = creat_tokenier(df)
    dataframe.to_csv('check.csv', index=False)
#-------------just for the test-------------