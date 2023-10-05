import torch
import pandas as pd
import torch.nn as nn
from vit_model import  vit_base_patch16_224_in21k as create_vit
import torch.nn.functional as F
from utils import creat_tokenier,splite_data,Tokenizer
from my_dataset import MyDataset
from torch.utils.data import DataLoader
image_folder = 'textmodel/images_training'
label_folder = 'processed_data.csv'
df = pd.read_csv(label_folder)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Image_encoder(nn.Module):
    def __init__(self, dense_dim, dropout_rate, vit_weight):
        super(Image_encoder, self).__init__()
        self.dense_dim = dense_dim
        self.vit = create_vit()
        
        if vit_weight:
            loaded_state_dict = torch.load(vit_weight, map_location='cpu')
            model_state_dict = self.vit.state_dict()

            # Filter out unnecessary keys
            loaded_state_dict = {k: v for k, v in loaded_state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}

            # Overwrite entries in the existing state dict
            model_state_dict.update(loaded_state_dict)

            # Load the new state dict
            self.vit.load_state_dict(model_state_dict)

        self.fc = nn.Linear(768, dense_dim)  # using dense layer to change the number of vector
        self.bn = nn.BatchNorm1d(dense_dim)
        self.dropout = nn.Dropout(dropout_rate)

        for param in self.vit.parameters():
            param.requires_grad = False

    def forward(self, data):
        op = self.vit.forward_features(data)  # op shape: (None,197,768)
        op = self.fc(op)  # op shape: [N, 197, 512]
        # check the shape of output
        return op

    def encoder(self, image1, image2, dropout_rate):
        """
        Takes image1,image2,
        gets the final encoded vector of these images
        """
        # image1 and image2
        bkfeat1 = self.forward(image1)  # shape: (None, 197, 768)
        bkfeat2 = self.forward(image2)  # shape: (None, 197, 768)

        # combining image1 and image2
        concat = torch.cat([bkfeat1, bkfeat2], dim=1)  # concatenating through the second axis shape: (None, 392, 768)
        concat = self.bn(concat.permute(0, 2, 1)).permute(0, 2, 1)
        # concat = concat.permute(0, 2, 1).permute(0, 2, 1)

        dropout_output = self.dropout(concat)
        return dropout_output

class GlobalAttention(nn.Module):
    """
    Calculate global attention
    """

    def __init__(self, dense_dim):
        super(GlobalAttention, self).__init__()
        # Initialize variables needed for Concat score function here
        self.W1 = nn.Linear(dense_dim, dense_dim)  # weight matrix of shape enc_units*dense_dim
        self.W2 = nn.Linear(dense_dim, dense_dim)  # weight matrix of shape dec_units*dense_dim
        self.V = nn.Linear(dense_dim, 1)  # weight matrix of shape dense_dim*1
        nn.init.normal_(self.V.weight, std=0.01)
        nn.init.normal_(self.W1.weight, std=0.01)

        nn.init.normal_(self.W2.weight, std=0.01)

    def forward(self, encoder_output,decoder_h):  # here the encoded output will be the concatted image bk features shape: (None,196,768)
        # Expand the decoder hidden shatch the encoder output shape
        # decoder_h = decoder_h[:, -1, :].unsqueeze(0)
        decoder_h = decoder_h.unsqueeze(1)#permute(1, 0, 2)#unsqueeze(1)
        tanh_input = self.W1(encoder_output) + self.W2(decoder_h)  # output shape: batch_size*197*dense_dim
        tanh_output = torch.tanh(tanh_input)
        # tanh_output = tanh_input
        # print(tanh_output.size())
        # print(self.V(tanh_output).size())
        # exit()
        attention_weights = F.softmax(self.V(tanh_output), dim=1)  # shape= batch_size*196*1 getting attention alphas
        # print(attention_weights.shape, attention_weights[0,...])
        # Calculate the context vector
        op = attention_weights * encoder_output

        context_vector = torch.sum(op,dim=1)

        return context_vector, attention_weights

