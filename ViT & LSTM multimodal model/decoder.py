import torch
from torch import nn
from encoder import GlobalAttention
import numpy as np

# #add dropout layer
# class OneStepDecoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, max_pad, dense_dim, dropout_rate):
#         super(OneStepDecoder, self).__init__()
#         self.dense_dim = dense_dim
#         self.embedding = nn.Embedding(num_embeddings=vocab_size +1, embedding_dim=embedding_dim, padding_idx=0)
#         self.gru = nn.GRU(input_size=embedding_dim +dense_dim, hidden_size=dense_dim, batch_first=True)
#         self.gru_dropout = nn.Dropout(dropout_rate)  # New dropout layer
#         self.attention = GlobalAttention(dense_dim=dense_dim)
#         self.fc = nn.Linear(dense_dim, vocab_size +1)
#         self.fc_dropout = nn.Dropout(dropout_rate)  # New dropout layer
#         self.softmax = nn.Softmax(dim=2)
#
#     def forward(self, input_to_decoder, encoder_output, decoder_h):
#         if (input_to_decoder >= self.embedding.num_embeddings).any():
#             raise ValueError("Some indices are out of range for the embedding layer!")
#
#         embedding_op = self.embedding(input_to_decoder)
#         context_vector, attention_weights = self.attention(encoder_output, decoder_h)
#         context_vector_time_axis = context_vector.unsqueeze(1)
#         concat_input = torch.cat((context_vector_time_axis, embedding_op), dim=2)
#         output, decoder_h = self.gru(concat_input, decoder_h)
#         output = self.gru_dropout(output)  # Apply dropout
#         output = self.fc(output)
#         output = self.fc_dropout(output)  # Apply dropout
#         output = self.softmax(output)
#
#         return output, decoder_h, attention_weights

class OneStepDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_pad, dense_dim):
        super(OneStepDecoder, self).__init__()
        # glove_vectors = self.load_glove_vectors()

        embedding_dim = 300
        self.dense_dim = dense_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(torch.from_numpy(np.load('glove_300_weight.npy')))
        self.embedding.weight.requires_grad = True

        # self.embedding = nn.Linear(300, embedding_dim)

        self.LSTM = nn.GRU(embedding_dim + dense_dim, dense_dim, batch_first=True)
        self.attention = GlobalAttention(dense_dim)
        self.concat = torch.cat
        self.dense = nn.Linear(dense_dim, dense_dim)
        self.final = nn.Linear(dense_dim, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, input_to_decoder, encoder_output, decoder_h):
        embedding_op = self.embedding(input_to_decoder)
        # print(embedding_op.size())
        #print('embedding_op shape: ',embedding_op.shape)
        context_vector, attention_weights = self.attention(encoder_output, decoder_h)
        context_vector_time_axis = context_vector.unsqueeze(1)
        concat_input = self.concat((context_vector_time_axis, embedding_op.unsqueeze(1)), dim=-1)#torch.Size([128, 1, 812])
        #print('concat_input in onetimedecoder shape: ',concat_input.shape)
        output, decoder_h = self.LSTM(concat_input, decoder_h.unsqueeze(0))
        # attention_weights = None
        # output, decoder_h = self.LSTM(embedding_op.unsqueeze(1), decoder_h.unsqueeze(0))

        output = self.dense(output)
        output = self.relu(output)  # ReLU activation
        output = self.final(output)
        #print('output +vocadsize: ',output.shape)
        return output, decoder_h.squeeze(0), attention_weights


class Decoder(nn.Module):
    def __init__(self, max_pad, embedding_dim, dense_dim, vocab_size):
        super(Decoder, self).__init__()
        self.one_step_decoder = OneStepDecoder(vocab_size, embedding_dim, max_pad, dense_dim)
        self.max_pad = max_pad

        # self.emb = nn.Linear(300, 512)
        self.num_layers = 2
        self.dense_dim = dense_dim
        self.gru = nn.GRU(input_size=300, hidden_size=dense_dim, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(dense_dim, vocab_size)
        # self.fc = nn.Linear(300, vocab_size)

        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        # torch.nn.init.normal_(self.gru.weight_ih_l0, mean=0.0, std=0.01)
        # torch.nn.init.normal_(self.gru.weight_hh_l0, mean=0.0, std=0.01)
        #
        torch.nn.init.zeros_(self.gru.bias_ih_l0)
        torch.nn.init.zeros_(self.gru.bias_hh_l0)

    def forward(self, encoder_output, caption):
        decoder_h = torch.zeros(encoder_output.size(0), encoder_output.size(2)).to(encoder_output.device)
        output_array = []
        #print('caption shape: ',caption.size())

        for timestep in range(self.max_pad):
            output, decoder_h, attention_weights = self.one_step_decoder(caption[:, timestep], encoder_output, decoder_h)
            output_array.append(output)
        #print('After timestep output shape: ', output.shape)
        #print('After timestep decoder_h shape: ', decoder_h.shape)
        output_array = torch.stack(output_array, dim=1)
       # print('After timestep output_array shape: ', output_array.shape)
        # output_array = self.fc(output_array)
        # print(output_array.size())
        # caption = self.emb(caption)
        # out, hd = self.gru(caption)  # （B, T, hidden_size）
        # print(out.size(), hd.size())
        # exit()
        # print(out.size())
        # exit()
        # output_array = self.fc(out)
        # output_array = self.fc(caption)


        return output_array
