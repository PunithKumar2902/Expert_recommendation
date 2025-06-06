import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

import torch.nn.functional as F

from collections import OrderedDict

import os
from Src.Utils import *

from Src.hGCN.hGCN import hGCNEncoder


class Ranking_model(nn.Module):

    def __init__(self, embedding_dim,kernel_1,cnn_channel):
        super(Ranking_model, self).__init__()

        self.emb_dim = embedding_dim
        self.out_channel = cnn_channel
        self.kernel_1 = kernel_1

        self.convnet1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, self.out_channel, kernel_size=(1, kernel_1))),
            ('relu1', nn.ReLU())
        ]))

        self.convnet2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(1, self.out_channel, kernel_size=(2, kernel_1))),
            ('relu2', nn.ReLU())
        ]))

        self.conv_features_size = self.emb_dim-self.kernel_1+1
        self.kernel_2 = self.conv_features_size//2+1

        self.convnet3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(1, 1, kernel_size=(3, self.kernel_2))),
            ('relu3', nn.ReLU())
        ]))

        self.features_for_NN = self.conv_features_size - self.kernel_2+1

        self.fc_new_1 = nn.Linear(self.features_for_NN, self.features_for_NN//2)
        self.fc_new_2 = nn.Linear(self.features_for_NN//2, 4)
        self.fc_new_3 = nn.Linear(4, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for Linear layers
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, users, top_user, question, uids, neg_users):

        B, U, D = users.size()

        # Expand top_user and question to match shape
        top_user_expanded = top_user.unsqueeze(1).expand(-1, U, -1)  # [B, U, D]
        question_expanded = question.unsqueeze(1).expand(-1, U, -1)  # [B, U, D]

        # Create user-question pairs
        low_rank_mat = torch.stack((users, question_expanded), dim=2).unsqueeze(2)  # [B, U, 1, 2, D]
        high_rank_mat = torch.stack((top_user_expanded, question_expanded), dim=2).unsqueeze(2)  # [B, U, 1, 2, D]

        # Reshape for CNN: merge batch and user dimensions
        low_rank_mat = low_rank_mat.view(-1, 1, 2, D)    # [(B*U), 1, 2, D]
        high_rank_mat = high_rank_mat.view(-1, 1, 2, D)  # [(B*U), 1, 2, D]

        # Conv layers
        def get_score(mat):
            x = torch.cat([self.convnet1(mat), self.convnet2(mat)], dim=2)
            x = self.convnet3(x)
            x = x.squeeze()  # [(B*U), T]
            x = self.fc_new_3(self.fc_new_2(self.fc_new_1(x)))
            return x.squeeze()  # [(B*U)]

        low_scores = get_score(low_rank_mat).view(B,U)
        high_scores = get_score(high_rank_mat).view(B,U)

        # Mask out padding
        valid_mask = (uids != 0)  # [B*U], True for valid users
        valid_mask = valid_mask.to(low_scores.device)

        # Compute loss only for valid users
        pairwise_diff = low_scores-high_scores

        # print("pairwise :",pairwise_diff.size())
        # print("valid ",valid_mask.size())
        pairwise_loss = torch.relu(pairwise_diff.masked_select(valid_mask)).mean()

        # q_neg_exp = question.unsqueeze(1).expand(-1, neg_users.size(1), -1)  # [B, 10, D]
        # neg_rank_mat = torch.stack((neg_users, q_neg_exp), dim=2).unsqueeze(2)  # [B, 10, 1, 2, D]
        # neg_rank_mat = neg_rank_mat.view(-1, 1, 2, D)  # [B*10, 1, 2, D]

        # neg_scores = get_score(neg_rank_mat).view(B,neg_users.size()[1])

        ## === 5. Contrastive Loss Term ===
        ## valid_mask = valid_mask.view(-1)

        # print("=================================================================")
        # print("low score : ",low_scores.size())
        # print("valid mask : ",valid_mask.size())
        # print("=================================================================")

        # min_low_score = low_scores.masked_fill(~valid_mask, float('inf')).min(dim=1).values  # [B]
        # max_neg_score = neg_scores.max(dim=1).values  # [B]

        # contrastive_loss = torch.nn.functional.relu(max_neg_score - min_low_score).mean()

        # total_loss = pairwise_loss + contrastive_loss

        return pairwise_loss
        # return total_loss
    
    def test(self,users, question):
        
        b_s = users.size()[0]
        questions  =  question.repeat(b_s,1)
        rank_mat = torch.stack((users,questions),dim=1).unsqueeze(1)

        score = torch.cat([
            self.convnet1(rank_mat)
            , self.convnet2(rank_mat)]
            , dim=2)    

        score = self.convnet3(score)

        score = self.fc_new_3(self.fc_new_2(
            self.fc_new_1(score).squeeze()))

        return score

class Encoder(nn.Module):
    def __init__(
            self,
            num_types, d_model, n_layers, n_head, dropout):
        super().__init__()
        self.d_model = d_model

    
        directory_path = 'user_pre_training/data/{dataset}/'.format(dataset=C.DATASET)
        tag_matrix_file = 'tag_matrix.npy'
        if not os.path.exists(directory_path + tag_matrix_file):
            print('tag_matrix is not found, Please insert it ...')
            return
        print('\nLoading ', directory_path + tag_matrix_file, '...')
        self.ui_adj = np.load(directory_path + tag_matrix_file)
        self.ui_adj = sp.csr_matrix(self.ui_adj)
        print('\nComputing adj matrix ...')
        self.ui_adj = torch.tensor(self.normalize_graph_mat(self.ui_adj).toarray(), device='cuda:0')

        self.layer_stack = nn.ModuleList([
            hGCNEncoder(d_model, n_head)
            for _ in range(n_layers)])


    def forward(self, event_type, enc_output, slf_attn_mask, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # get individual adj
        adj = torch.zeros((event_type.size(0), event_type.size(1), event_type.size(1)), device='cuda:0')
        for i, e in enumerate(event_type):
            # Thanks to Lin Fang for reminding me to correct a mistake here.
            adj[i] = self.ui_adj[e-1, :][:, e-1]
            # performance can be enhanced by adding the element in the diagonal of the normalized adjacency matrix.
            adj[i] += self.ui_adj[e-1, e-1]

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, adj, event_type)

        return enc_output.mean(1)

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        rowsum[rowsum==0] = 1e-9
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat


class Decoder(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.dropout = nn.Dropout(0.5)
        self.temperature = 512 ** 0.5

        self.conv = torch.nn.Conv2d(1, 1, (3, 3), padding=1, padding_mode='zeros')
        self.conv3 = torch.nn.Conv2d(1, 1, (700, 1))

    def forward(self, user_embeddings, embeddings, enc_output, slf_attn_mask):
        outputs = []

        outputs.append(user_embeddings)

        # seq1 implicit
        attn = torch.matmul(enc_output / self.temperature, enc_output.transpose(1, 2))
        attn = self.dropout(torch.tanh(attn)) * slf_attn_mask
        seq1_implicit = torch.matmul(attn, enc_output)
        seq1_implicit = (seq1_implicit.mean(1))
        outputs.append(seq1_implicit/2)

        # seq2 implicit
        seq2_implicit = self.conv(enc_output.unsqueeze(1))
        seq2_implicit = self.conv3(seq2_implicit)
        seq2_implicit = (seq2_implicit.squeeze(1).squeeze(1))
        outputs.append(seq2_implicit*2)

        outputs = torch.stack(outputs, dim=0).sum(0)
        out = torch.tanh(outputs)
        return out


class Embed_Model(nn.Module):
    def __init__(
            self,  num_types, d_model=256, n_layers=4, n_head=4, dropout=0.1, device=0):
        super(Embed_Model, self).__init__()

        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=C.PAD)  # dding 0
        self.encoder = Encoder(
            num_types=num_types, d_model=d_model,
            n_layers=n_layers, n_head=n_head, dropout=dropout)

        self.num_types = num_types
        self.decoder = Decoder(d_model, num_types)

    def forward(self, event_type):
        slf_attn_mask_subseq = get_subsequent_mask(event_type)  # M * L * L
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)  # M x lq x lk
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        non_pad_mask = get_non_pad_mask(event_type)

        # (K M)  event_emb: Embedding
        enc_output = self.event_emb(event_type)

        user_embeddings = self.encoder(event_type, enc_output, slf_attn_mask, non_pad_mask)  # H(j,:)

        prediction = self.decoder(user_embeddings, self.event_emb.weight, enc_output, slf_attn_mask)

        return prediction, user_embeddings
